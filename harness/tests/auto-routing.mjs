import test from 'node:test'
import assert from 'node:assert/strict'
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {runAuto,advanceRouting,routingState} from '../agents/auto.mjs'

test('coordinator fallback persists exact models, efforts and technical evidence',async t=>{
  const directory=await mkdtemp(join(tmpdir(),'dsh-auto-')),previous=process.env.DSH_HOME
  process.env.DSH_HOME=directory
  t.after(async()=>{if(previous===undefined)delete process.env.DSH_HOME;else process.env.DSH_HOME=previous;await rm(directory,{recursive:true,force:true})})
  const calls=[]
  const result=await runAuto({sessionId:'test-parent'},async(route,state)=>{
    calls.push(route)
    if(route.service==='cursor')throw Error('Cursor quota exhausted')
    assert(state.history[0].reason.includes('quota'))
    return 'astra success'
  })
  assert.equal(result,'astra success')
  assert.deepEqual(calls.map(c=>c.service),['cursor','codex'])
  assert.deepEqual(calls[1].options,{model:'gpt-6.2-astra',effort:'high',subscription:true})
  await runAuto({sessionId:'test-parent'},async route=>assert.equal(route.service,'codex'))
  await advanceRouting('test-parent','Two approaches failed',['compiler output','test failure'])
  await runAuto({sessionId:'test-parent'},async(route,state)=>{
    assert.deepEqual(route.options,{model:'claude-opus-5-5',effort:'high'})
    assert.deepEqual(state.history[1].evidence,['compiler output','test failure'])
  })
  await assert.rejects(runAuto({sessionId:'test-parent'},async()=>{throw Error('still unresolved')}),/全段階/)
  assert.equal((await routingState('test-parent')).exhausted,true)
  await assert.rejects(runAuto({sessionId:'test-parent'},()=>assert.fail('must not launch')),/未解決/)
})

test('cancellation, auxiliary failures and refusals never trigger another model',async()=>{
  for(const options of [{purpose:'title'},{signal:AbortSignal.abort()}]) {
    let called=0
    await assert.rejects(runAuto(options,async()=>{called++;throw Error('failure')}))
    assert.equal(called,options.signal?0:1)
  }
  for(const message of ['Model refusal','Permission denied','Safety policy']) {
    let called=0
    await assert.rejects(runAuto({},async()=>{called++;throw Error(message)}))
    assert.equal(called,1)
  }
})

test('legacy Astra resumes once and never repeats the previously failed Opus',async t=>{
  const directory=await mkdtemp(join(tmpdir(),'dsh-auto-legacy-')),previous=process.env.DSH_HOME
  process.env.DSH_HOME=directory
  t.after(async()=>{if(previous===undefined)delete process.env.DSH_HOME;else process.env.DSH_HOME=previous;await rm(directory,{recursive:true,force:true})})
  await mkdir(join(directory,'routing'))
  const history=[{route:'Cursor',reason:'legacy quota',evidence:['cursor receipt']},
    {route:'Claude Opus 5.5',reason:'legacy technical failure',evidence:['opus receipt']}]
  await writeFile(join(directory,'routing/old-astra.json'),JSON.stringify({stage:2,history}))
  const calls=[]
  await assert.rejects(runAuto({sessionId:'old-astra'},async(route,state)=>{
    calls.push(route.service)
    assert.deepEqual(state.history,history)
    throw Error('Astra remains unresolved')
  }),/全段階/)
  assert.deepEqual(calls,['codex'])
  const saved=await routingState('old-astra')
  assert.equal(saved.exhausted,true)
  assert.deepEqual(saved.history.slice(0,2),history)
  assert.equal(saved.history[2].route,'Codex Astra 6.2')
})

test('legacy Opus resumes its actual model then tries previously untried Astra with history intact',async t=>{
  const directory=await mkdtemp(join(tmpdir(),'dsh-auto-legacy-')),previous=process.env.DSH_HOME
  process.env.DSH_HOME=directory
  t.after(async()=>{if(previous===undefined)delete process.env.DSH_HOME;else process.env.DSH_HOME=previous;await rm(directory,{recursive:true,force:true})})
  await mkdir(join(directory,'routing'))
  const history=[{route:'Cursor',reason:'legacy quota',evidence:['cursor receipt']}]
  await writeFile(join(directory,'routing/old-opus.json'),JSON.stringify({stage:1,history}))
  const calls=[]
  const answer=await runAuto({sessionId:'old-opus'},async(route,state)=>{
    calls.push(route.service)
    assert.deepEqual(state.history.slice(0,1),history)
    if(route.service==='claude')throw Error('Opus remains unresolved')
    assert.equal(state.history[1].route,'Claude Opus 5.5')
    assert.equal(state.history[1].reason,'Opus remains unresolved')
    return 'resolved by Astra'
  })
  assert.equal(answer,'resolved by Astra')
  assert.deepEqual(calls,['claude','codex'])
  await assert.rejects(runAuto({sessionId:'old-opus'},async()=>{throw Error('later Astra failure')}),/全段階/)
  const saved=await routingState('old-opus')
  assert.equal(saved.exhausted,true)
  assert.deepEqual(saved.history.map(h=>h.route),['Cursor','Claude Opus 5.5','Codex Astra 6.2'])
})
