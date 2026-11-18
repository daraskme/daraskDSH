import test from 'node:test'
import assert from 'node:assert/strict'
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises'
import {join} from 'node:path'
import {tmpdir} from 'node:os'
import {TaskQueue} from '../plugins/task-router/queue.mjs'
import {validatePlan,conflicts,routes,routingVersion} from '../plugins/task-router/policy.mjs'
import {capacityConfig,pressure} from '../plugins/task-router/capacity.mjs'

const task=(id,extra={})=>({id,phase:'analysis',title:id,prompt:'Inspect '+id,acceptance:['Proof for '+id],dependsOn:[],reads:[],writes:[id],...extra})
const plan=tasks=>({version:1,objective:'Test analysis and port',inventoryComplete:true,sources:['checkpoint'],phases:[{id:'analysis',title:'Analysis'}],tasks})
const report=(status='resolved')=>({status,summary:'Evidence summary',evidence:['fixture test passed'],attempts:['Approach one','Approach two'],blockers:status==='resolved'?[]:['unresolved condition'],changedFiles:[]})
const result=status=>({stopReason:'completed',structured:report(status)})
const parent={session:{header:{id:'parent-test'}}}
async function until(check) {for(let i=0;i<250;i++){if(await check())return;await new Promise(r=>setTimeout(r,10))}throw Error('Timed out')}
async function fixture(t,tasks) {
  const root=await mkdtemp(join(tmpdir(),'dsh-queue-')),calls=[],errors=[]
  await writeFile(join(root,'plan.json'),JSON.stringify(plan(tasks)))
  const config={root,stateRoot:join(root,'state'),capacity:{adaptive:false,total:3,ceilings:{cursor:2,grok:1,codex:1},initial:{cursor:2,grok:1,codex:1}},onError:e=>errors.push(e.message),start:async spec=>{
    let finish;const settled=new Promise(r=>finish=r)
    const call={...spec,finish,childId:'child-'+calls.length};calls.push(call)
    const abort=()=>finish({stopReason:'aborted'})
    spec.signal.addEventListener('abort',abort,{once:true});if(spec.signal.aborted)abort()
    return {childId:call.childId,result:settled,dispose:async()=>abort()}
  }}
  const queue=new TaskQueue(config)
  t.after(async()=>{await queue.close();assert.deepEqual(errors,[]);await rm(root,{recursive:true,force:true})})
  await queue.importPlan('plan.json')
  return {root,queue,calls,config}
}
test('plan validation requires an inventory and rejects missing dependencies/cycles',()=>{
  assert.throws(()=>validatePlan({...plan([task('a')]),inventoryComplete:false}),/全体計画/)
  assert.throws(()=>validatePlan(plan([task('a',{dependsOn:['missing']})])),/存在しない/)
  assert.throws(()=>validatePlan(plan([task('a',{dependsOn:['b']}),task('b',{dependsOn:['a']})])),/循環/)
  assert(conflicts(task('a',{writes:['src']}),task('b',{reads:['src/file.c']})))
  assert(!conflicts(task('a',{writes:[],reads:['src']}),task('b',{writes:[],reads:['src']})))
  assert.equal(routes[1].options.model,'gpt-6.2-astra');assert.equal(routes[1].options.subscription,true)
  assert.equal(routes[2].options.model,'claude-opus-5-5')
})
test('maximum budget is local and configurable; quota and concurrent limits differ',()=>{
  assert.equal(capacityConfig({}, {cpus:24,memory:108*2**30}).total,48)
  assert.equal(capacityConfig({}, {cpus:24,memory:8*2**30}).total,4)
  assert.equal(capacityConfig({DSH_TASK_MAX_PARALLEL:'12'}).total,12)
  assert.throws(()=>capacityConfig({DSH_TASK_MAX_PARALLEL:'0'}))
  assert.equal(capacityConfig({}).ceilings.codex,1)
  assert.equal(capacityConfig({}).initial.codex,1)
  assert.equal(pressure('Free quota exhausted'),'quota')
  assert.equal(pressure('429 Too many requests'),'concurrency')
})
test('Cursor/Grok-only pool, dependency/review gates and declared file conflicts',async t=>{
  const {queue,calls}=await fixture(t,[task('a'),task('b'),task('c'),task('d'),task('overlap',{reads:['a']}),task('dependent',{dependsOn:['a']})])
  await queue.run(parent);await until(()=>calls.length===3)
  assert.deepEqual(calls.map(c=>c.task.id),['a','b','c'])
  assert.deepEqual(calls.map(c=>c.route.service),['cursor','grok','cursor'])
  assert.deepEqual((await queue.status()).basePool.map(r=>r.service),['cursor','grok'])
  calls[0].finish(result('resolved'));await until(()=>calls.length===4)
  assert.equal(calls[3].task.id,'d')
  assert.equal((await queue.status()).counts.review,1)
  calls[1].finish(result('resolved'));await until(async()=>(await queue.status()).counts.review===2)
  assert.equal(calls.length,4)
  assert(!calls.some(c=>c.task.id==='dependent'))
  await queue.review('a',true,['Independent test verified'])
  await until(()=>calls.some(c=>c.task.id==='overlap'))
  calls[2].finish(result('resolved'));await until(()=>calls.some(c=>c.task.id==='dependent'))
})

test('maximum mode grows provider lanes and backs off after a concurrency error',async t=>{
  const {queue,calls}=await fixture(t,Array.from({length:20},(_,i)=>task('task-'+i)))
  queue.capacity={adaptive:true,total:8,ceilings:{cursor:8,grok:8,codex:1},initial:{cursor:4,grok:4,codex:1}}
  queue.state.lanes={}
  await queue.run(parent);await until(()=>calls.length===8)
  const cursor=calls.find(c=>c.route.service==='cursor')
  cursor.finish(result('resolved'))
  await until(()=>calls.length===9)
  assert.equal(queue.active.size,8)
  assert.equal((await queue.status()).basePool.find(p=>p.service==='cursor').limit,5)
  calls.find(c=>c!==cursor&&c.route.service==='cursor').finish({stopReason:'error',diagnostic:'429: Too many concurrent requests'})
  await until(async()=>(await queue.status()).basePool.find(p=>p.service==='cursor').limit===2)
  assert(queue.active.size<=8)
  assert.equal((await queue.status()).basePool.find(p=>p.service==='cursor').lastPressure.kind,'concurrency')
})
test('unresolved work escalates to pinned models with evidence and stops after Opus',async t=>{
  const {queue,calls}=await fixture(t,[task('a')])
  await queue.run(parent);await until(()=>calls.length===1)
  calls[0].finish(result('unresolved'));await until(()=>calls.length===2)
  assert.equal(calls[1].route.options.model,'gpt-6.2-astra')
  assert(calls[1].prompt.includes('Approach one'))
  calls[1].finish({stopReason:'error',diagnostic:'temporary model failure'});await until(()=>calls.length===3)
  assert.equal(calls[2].route.options.model,'claude-opus-5-5')
  assert(calls[2].prompt.includes('temporary model failure'))
  calls[2].finish(result('unresolved'));await until(async()=>(await queue.status()).counts.unresolved===1)
  const status=await queue.status();assert.equal(calls.length,3)
  const saved=await readFile(status.unresolvedPath,'utf8');assert(saved.includes('Claude Opus 5.5'));assert(saved.includes('unresolved condition'))
})
test('fallback workers run one at a time, pauses never escalate, and state survives reopening',async t=>{
  const {queue,calls,config}=await fixture(t,[task('a'),task('b')])
  await queue.run(parent);await until(()=>calls.length===2)
  calls[0].finish(result('unresolved'));calls[1].finish(result('unresolved'))
  await until(()=>calls.length===3)
  assert.equal(calls[2].route.service,'codex')
  await queue.pause();assert.equal(calls.length,3)
  const paused=await queue.status();assert.equal(paused.enabled,false);assert.equal(paused.counts.interrupted,1)
  await queue.close()
  const restored=new TaskQueue(config)
  const state=await restored.status();assert.equal(state.enabled,false);assert.equal(state.counts.interrupted,1)
  await restored.run(parent);await until(()=>calls.length===4)
  assert.equal(calls[3].route.service,'codex')
  await restored.close()
})
test('external blockers and changed files outside scope require parent intervention',async t=>{
  const {queue,calls}=await fixture(t,[task('a'),task('b')])
  await queue.run(parent);await until(()=>calls.length===2)
  calls[0].finish(result('external_blocked'))
  calls[1].finish({stopReason:'completed',structured:{...report(),changedFiles:['outside/file']}})
  await until(async()=>(await queue.status()).counts.unresolved===2)
  assert.equal(calls.length,2)
})
test('exhausted normal pool still tries Astra before Opus and respects separate fallback quota',async t=>{
  const {queue,calls}=await fixture(t,Array.from({length:7},(_,i)=>task('q-'+i)))
  await queue.run(parent);await until(()=>calls.length===3)
  for(const call of calls.slice(0,3))call.finish({stopReason:'error',diagnostic:'Free quota exhausted'})
  await until(()=>calls.length===4)
  assert.equal(calls[3].route.service,'codex')
  assert.equal(queue.state.lanes.codex.disabled,false)
  assert.equal((await queue.status()).basePool.filter(p=>p.disabled).length,2)
  calls[3].finish({stopReason:'error',diagnostic:'Subscription quota exhausted'})
  for(let i=0;i<7;i++) {
    await until(()=>calls.some(c=>c.route.service==='claude'&&!c.testFinished))
    const next=calls.find(c=>c.route.service==='claude'&&!c.testFinished)
    next.testFinished=true;next.finish(result('resolved'))
  }
  await until(async()=>(await queue.status()).counts.review===7)
  assert.equal(calls.filter(c=>['cursor','grok'].includes(c.route.service)).length,3)
  assert.equal(calls.filter(c=>c.route.service==='codex').length,1)
  assert.equal(calls.filter(c=>c.route.service==='claude').length,7)
})

test('Astra fallback quota does not disable normal Cursor/Grok dispatch',async t=>{
  const {queue,calls}=await fixture(t,[task('a'),task('b'),task('c'),task('later')])
  await queue.run(parent);await until(()=>calls.length===3)
  calls[0].finish(result('unresolved'))
  await until(()=>calls.length===4)
  assert.equal(calls[3].route.service,'codex')
  calls[3].finish({stopReason:'error',diagnostic:'Subscription quota exhausted'})
  await until(()=>calls.length===5)
  assert.equal(calls[4].route.service,'claude')
  assert((await queue.status()).basePool.every(lane=>!lane.disabled))
  calls[1].finish(result('resolved'))
  await until(()=>calls.length===6)
  assert.equal(calls[5].task.id,'later')
  assert(['cursor','grok'].includes(calls[5].route.service))
})

test('another host cannot acquire the same workspace queue',async t=>{
  const {queue,config}=await fixture(t,[task('a')])
  const other=new TaskQueue(config)
  await assert.rejects(other.status(),/別のDSH/)
  await other.close().catch(()=>{})
  assert.equal((await queue.status()).counts.pending,1)
})

test('a stale running record after a crash recovers as interrupted, never successful',async t=>{
  const {queue,config}=await fixture(t,[task('a')])
  const path=(await queue.status()).statePath
  await queue.close()
  const saved=JSON.parse(await readFile(path,'utf8'))
  Object.assign(saved.tasks[0],{status:'running',baseService:'cursor',attempts:[{route:'Cursor',startedAt:'fixture'}]})
  saved.enabled=true
  await writeFile(path,JSON.stringify(saved))
  const restored=new TaskQueue(config)
  const state=await restored.status()
  assert.equal(state.enabled,false)
  assert.equal(state.counts.interrupted,1)
  assert.equal(state.tasks[0].lastAttempt.stopReason,'interrupted')
  await restored.close()
})

test('removed Devin assignments resume in the current pool without losing previous attempts',async t=>{
  const {queue,config,calls}=await fixture(t,[task('a')])
  const path=(await queue.status()).statePath
  await queue.close()
  const saved=JSON.parse(await readFile(path,'utf8'))
  Object.assign(saved.tasks[0],{status:'interrupted',baseService:'devin-free',baseOptions:{model:'old-devin'},
    attempts:[{route:'Devin（無料のみ）',stopReason:'aborted'}]})
  saved.lanes['devin-free']={limit:3}
  await writeFile(path,JSON.stringify(saved))
  const restored=new TaskQueue(config)
  await restored.run(parent);await until(()=>calls.length===1)
  assert.equal(calls[0].route.service,'cursor')
  assert.equal(calls[0].task.baseOptions,undefined)
  assert.equal(calls[0].task.attempts[0].route,'Devin（無料のみ）')
  assert.equal(calls[0].task.reassignments[0].from,'devin-free')
  assert(!(await restored.status()).basePool.some(p=>p.service==='devin-free'))
  await restored.close()
})

test('failed independent review passes evidence to the next model',async t=>{
  const {queue,calls}=await fixture(t,[task('a')])
  await queue.run(parent);await until(()=>calls.length===1)
  calls[0].finish(result('resolved'))
  await until(async()=>(await queue.status()).counts.review===1)
  await queue.review('a',false,['Regression test failed'],'Actual result does not meet acceptance')
  await until(()=>calls.length===2)
  assert.equal(calls[1].route.service,'codex')
  assert(calls[1].prompt.includes('Actual result does not meet acceptance'))
})

for(const status of ['pending','interrupted'])test(`legacy normal Astra ${status} work is reassigned to Cursor/Grok with history retained`,async t=>{
  const {queue,config,calls}=await fixture(t,[task('legacy-astra')])
  const path=(await queue.status()).statePath
  await queue.close()
  const saved=JSON.parse(await readFile(path,'utf8'))
  const attempts=status==='pending'?[]:[{route:'Codex Astra 6.2',stopReason:'aborted',childId:'legacy-child'}]
  Object.assign(saved.tasks[0],{stage:0,status,baseService:'codex',baseOptions:{model:'gpt-6.2-astra',effort:'high',subscription:true},attempts})
  await writeFile(path,JSON.stringify(saved))
  const restored=new TaskQueue(config)
  await restored.run(parent);await until(()=>calls.length===1)
  assert(['cursor','grok'].includes(calls[0].route.service))
  assert.deepEqual(calls[0].task.attempts.slice(0,attempts.length),attempts)
  assert.equal(calls[0].task.baseOptions,undefined)
  assert.equal(calls[0].task.reassignments[0].from,'codex')
  assert.deepEqual((await restored.status()).basePool.map(r=>r.service),['cursor','grok'])
  await restored.close()
})

test('legacy normal Astra review and accepted evidence retain the author; review rejection goes to Opus',async t=>{
  const {queue,config,calls}=await fixture(t,[task('review'),task('accepted')])
  const path=(await queue.status()).statePath
  await queue.close()
  const saved=JSON.parse(await readFile(path,'utf8'))
  for(const item of saved.tasks)Object.assign(item,{stage:0,status:item.id,baseService:'codex',attempts:[
    {route:'Codex Astra 6.2',stopReason:'completed',report:report(),childId:'legacy-'+item.id}]})
  const histories=saved.tasks.map(t=>structuredClone(t.attempts))
  await writeFile(path,JSON.stringify(saved))
  const restored=new TaskQueue(config)
  const state=await restored.status()
  assert.deepEqual(state.tasks.map(t=>t.route),['Codex Astra 6.2','Codex Astra 6.2'])
  assert.deepEqual(restored.state.tasks.map(t=>t.attempts),histories)
  assert.deepEqual(restored.state.tasks.map(t=>t.baseService),['codex','codex'])
  await restored.run(parent)
  assert.equal(calls.length,0)
  await restored.review('review',false,['Independent review failed'])
  await until(()=>calls.length===1)
  assert.equal(calls[0].route.service,'claude')
  calls[0].finish(result('unresolved'))
  await until(async()=>(await restored.status()).counts.unresolved===1)
  assert.equal(calls.length,1)
  assert.equal(restored.state.tasks.find(t=>t.id==='accepted').status,'accepted')
  await restored.close()
})

test('legacy fallback stages migrate by actual attempts and preserve review/interruption evidence',async t=>{
  const ids=['pending-opus','interrupted-opus','review-opus','pending-astra','interrupted-astra','review-astra']
  const {queue,config}=await fixture(t,ids.map(id=>task(id)))
  const path=(await queue.status()).statePath
  await queue.close()
  const saved=JSON.parse(await readFile(path,'utf8'))
  delete saved.routingVersion
  for(const item of saved.tasks) {
    const [status,target]=item.id.split('-')
    Object.assign(item,{status,stage:target==='opus'?1:2,baseService:'cursor',attempts:[{route:'Cursor',stopReason:'completed',report:report('unresolved')}]})
    if(target==='astra')item.attempts.push({route:'Claude Opus 5.5',stopReason:'completed',report:report('unresolved')})
    if(status!=='pending')item.attempts.push({route:target==='opus'?'Claude Opus 5.5':'Codex Astra 6.2',
      stopReason:status==='review'?'completed':'aborted',report:report(),childId:'legacy-'+item.id})
  }
  const histories=saved.tasks.map(t=>structuredClone(t.attempts))
  await writeFile(path,JSON.stringify(saved))
  const restored=new TaskQueue(config)
  await restored.status()
  assert.equal(restored.state.routingVersion,routingVersion)
  assert.deepEqual(restored.state.tasks.map(t=>t.stage),[1,2,2,1,1,1])
  assert.deepEqual(restored.state.tasks.map(t=>t.status),ids.map(id=>id.split('-')[0]))
  assert.deepEqual(restored.state.tasks.map(t=>t.attempts),histories)
  await restored.close()
})

test('a migrated Astra attempt does not repeat the already failed legacy Opus attempt',async t=>{
  const {queue,config,calls}=await fixture(t,[task('legacy')])
  const path=(await queue.status()).statePath
  await queue.close()
  const saved=JSON.parse(await readFile(path,'utf8'))
  delete saved.routingVersion
  Object.assign(saved.tasks[0],{stage:2,status:'pending',baseService:'cursor',attempts:[
    {route:'Cursor',stopReason:'completed',report:report('unresolved')},
    {route:'Claude Opus 5.5',stopReason:'completed',report:report('unresolved')} ]})
  await writeFile(path,JSON.stringify(saved))
  const restored=new TaskQueue(config)
  await restored.run(parent);await until(()=>calls.length===1)
  assert.equal(calls[0].route.service,'codex')
  calls[0].finish(result('unresolved'))
  await until(async()=>(await restored.status()).counts.unresolved===1)
  assert.equal(calls.length,1)
  assert.equal(restored.state.tasks[0].attempts.length,3)
  await restored.close()
})

for(const service of ['cursor','grok'])test(`pending work assigned to exhausted ${service} advances instead of waiting forever`,async t=>{
  const {queue,calls}=await fixture(t,[task('a'),task('b'),task('c'),task('resume-assigned')])
  const waiting=queue.state.tasks.find(t=>t.id==='resume-assigned')
  Object.assign(waiting,{baseService:service,attempts:[{route:service,stopReason:'aborted'}]})
  await queue.run(parent);await until(()=>calls.length===3)
  const failing=calls.find(c=>c.route.service===service)
  failing.finish({stopReason:'error',diagnostic:'Subscription quota exhausted'})
  for(const call of calls.slice(0,3))if(call!==failing)call.finish(result('resolved'))
  await until(()=>waiting.attempts.some(a=>a.stopReason==='unavailable'))
  assert(waiting.stage>0)
  await until(()=>calls.length===4)
  calls[3].finish(result('resolved'))
  await until(()=>calls.length===5)
  assert.equal(calls[4].task.id,'resume-assigned')
  assert.equal(calls[4].route.service,'codex')
  calls[4].finish(result('resolved'))
  await until(async()=>(await queue.status()).counts.review===4)
})
