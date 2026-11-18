import {mkdir,readFile,writeFile,rename,realpath,stat} from 'node:fs/promises'
import {join,relative,resolve} from 'node:path'
import {createHash,randomUUID} from 'node:crypto'
import {spawn} from 'node:child_process'
import {once} from 'node:events'
import {routes,baseRoutes,routingVersion,taskRoute,validatePlan,conflicts,workerPrompt,resultSchema,taskPath,stringList} from './policy.mjs'
import {capacityConfig,pressure} from './capacity.mjs'

const stamp=()=>new Date().toISOString()
const hash=text=>createHash('sha256').update(text).digest('hex')
const done=t=>t.status==='accepted'||t.status==='baseline'
const tried=(task,route)=>task.attempts.some(a=>a.route===route.label&&a.stopReason&&!['aborted','interrupted','unavailable'].includes(a.stopReason))
const inside=(root,path)=>{const rel=relative(root,path);return rel===''||(!rel.startsWith('../')&&rel!=='..'&&!rel.startsWith('/'))}

export class TaskQueue {
  constructor({root,stateRoot,start,onError=console.error,capacity=capacityConfig()}) {
    this.root=root;this.directory=join(stateRoot,hash(root).slice(0,24));this.start=start;this.onError=onError
    this.state=null;this.active=new Map();this.jobs=new Set();this.tail=Promise.resolve();this.parent=null;this.closed=false
    this.capacity=capacity
    this.ready=this.load()
  }
  serial(fn) {const p=this.tail.then(fn);this.tail=p.catch(()=>{});return p}
  async load() {
    await mkdir(this.directory,{recursive:true,mode:0o700})
    // flock is released by the OS even after a crash; another DSH host cannot
    // recover or launch this workspace's queue while its current owner is alive.
    this.lock=spawn('flock',['-n',join(this.directory,'runner.lock'),'/bin/sh','-c','printf "locked\\n"; cat >/dev/null'],{stdio:['pipe','pipe','pipe']})
    this.lock.stdin.on('error',()=>{})
    this.lockExit=once(this.lock,'exit').catch(()=>{})
    await new Promise((ok,fail)=>{
      this.lock.once('error',fail)
      this.lock.once('exit',()=>fail(Error('このワークスペースのキューは別のDSHが管理中です')))
      this.lock.stdout.once('data',()=>ok())
    })
    try {this.state=JSON.parse(await readFile(join(this.directory,'state.json'),'utf8'))}
    catch(e){if(e.code!=='ENOENT')throw e}
    if(this.state) {
      this.state.enabled=false
      this.initLanes()
      for(const task of this.state.tasks) if(task.status==='running') {
        task.status='interrupted';task.attempts.at(-1).stopReason='interrupted';task.attempts.at(-1).endedAt=stamp()
      }
      this.migrateRouting()
      this.releaseRemovedAssignments()
      await this.save()
    }
  }
  async save() {
    if(!this.state)return
    this.state.updatedAt=stamp()
    const target=join(this.directory,'state.json'),temp=target+'.'+randomUUID()
    await writeFile(temp,JSON.stringify(this.state,null,2)+'\n',{mode:0o600});await rename(temp,target)
    const lines=['# 未解決・確認待ちタスク','',this.state.objective,'','作業ルート: '+this.root,
      '計画: '+this.state.planPath,'参照: '+this.state.sources.join(', '),'記録時刻: '+this.state.updatedAt,'']
    for(const t of this.state.tasks.filter(t=>!done(t))) {
      lines.push('## '+t.id+' — '+t.title,'','状態: '+t.status,'フェーズ: '+t.phase,
        '依存: '+(t.dependsOn.join(', ')||'なし'),'作業: '+t.prompt,
        ...t.acceptance.map(v=>'- 受入条件: '+v),'参照範囲: '+(t.reads.join(', ')||'なし'),
        '変更範囲: '+(t.writes.join(', ')||'なし'),'')
      for(const a of t.attempts) lines.push('### '+a.route+' ('+(a.stopReason||'running')+')','',
        a.report?.summary || a.error || '実行中',...(a.report?.attempts||[]).map(v=>'- 試行: '+v),
        ...(a.report?.evidence||[]).map(v=>'- 証拠: '+v),...(a.report?.blockers||[]).map(v=>'- 未解決: '+v),
        ...(a.report?.changedFiles||[]).map(v=>'- 変更: '+v),a.childId?'DSH子セッション: '+a.childId:'','')
      for(const a of t.attempts)if(a.error||a.handoff)lines.push('- 停止・引き継ぎ理由: '+(a.error||a.handoff))
    }
    const report=join(this.directory,'UNRESOLVED.md'),tmp=report+'.'+randomUUID()
    await writeFile(tmp,lines.join('\n')+'\n',{mode:0o600});await rename(tmp,report)
  }
  async importPlan(path) {
    await this.ready
    return this.serial(async()=>{
      if(this.active.size)throw Error('実行中の計画は変更できません')
      const file=await realpath(resolve(this.root,path))
      if(!inside(this.root,file))throw Error('計画ファイルはワークスペース内に置いてください')
      if((await stat(file)).size>16*1024*1024)throw Error('計画ファイルが大きすぎます')
      const raw=await readFile(file,'utf8'),plan=validatePlan(JSON.parse(raw))
      const old=new Map(this.state?.tasks.map(t=>[t.id,t])||[])
      for(const t of plan.tasks) {
        const prior=old.get(t.id)
        if(prior && JSON.stringify(Object.fromEntries(Object.keys(t).map(k=>[k,prior[k]])))!==JSON.stringify(t))throw Error('既存タスクの定義を変える場合は新しいIDが必要です: '+t.id)
      }
      if([...old.keys()].some(id=>!plan.tasks.some(t=>t.id===id)))throw Error('既存タスクの記録は削除できません')
      this.state={...this.state,...plan,routingVersion,root:this.root,planPath:file,planHash:hash(raw),enabled:false,
        tasks:plan.tasks.map(t=>old.get(t.id)||{...t,status:t.baselineEvidence.length?'baseline':'pending',stage:0,attempts:[]})}
      this.initLanes()
      await this.save();return this.snapshot()
    })
  }
  snapshot(offset=0,limit=30) {
    if(!this.state)return {initialized:false,root:this.root}
    const counts={};for(const t of this.state.tasks)counts[t.status]=(counts[t.status]||0)+1
    return {initialized:true,root:this.root,enabled:this.state.enabled,objective:this.state.objective,
      basePool:baseRoutes.map(r=>({service:r.service,label:r.label,...this.state.lanes?.[r.service],ceiling:this.capacity.ceilings[r.service]})),
      routingVersion,fallback:routes.slice(1).map(r=>r.label),maxParallel:this.capacity.total,maxFallbackParallel:1,capacityMode:this.capacity.adaptive?'adaptive-maximum':'fixed',counts,total:this.state.tasks.length,
      statePath:join(this.directory,'state.json'),unresolvedPath:join(this.directory,'UNRESOLVED.md'),
      taskOrder:'review, unresolved, interrupted, running, pending, accepted, baseline',
      tasks:[...this.state.tasks].sort((a,b)=>['review','unresolved','interrupted','running','pending','accepted','baseline'].indexOf(a.status)-['review','unresolved','interrupted','running','pending','accepted','baseline'].indexOf(b.status)).slice(offset,offset+limit).map(t=>({id:t.id,phase:t.phase,title:t.title,status:t.status,
        route:taskRoute(t)?.label||t.baseService||'Cursor / Grokから自動割当',dependsOn:t.dependsOn,lastAttempt:t.attempts.at(-1)}))}
  }
  migrateRouting() {
    if(this.state.routingVersion===routingVersion)return
    if(this.state.routingVersion!==undefined)throw Error('未対応のタスク担当構成です: '+this.state.routingVersion)
    for(const task of this.state.tasks) {
      if(task.stage===0)continue
      const oldStage=task.stage
      // Prior queues used stage 1=Opus, stage 2=Astra. Never reinterpret an
      // interrupted execution or a report awaiting review as another model.
      if(task.status==='pending') {
        task.stage=routes.findIndex((r,i)=>i>0&&!tried(task,r))
        if(task.stage<0){task.stage=routes.length-1;task.status='unresolved'}
      } else task.stage=oldStage===1?2:1
      task.reassignments=[...task.reassignments||[],{fromStage:oldStage,toStage:task.stage,
        reason:'担当順をAstra 6.2→Opus 5.5へ更新（既存試行・確認待ちは保持）',time:stamp()}]
    }
    this.state.routingVersion=routingVersion
  }
  initLanes() {
    this.state.lanes??={}
    const laneRoutes=[...baseRoutes,routes[1]]
    for(const service of Object.keys(this.state.lanes))if(!laneRoutes.some(r=>r.service===service))delete this.state.lanes[service]
    for(const r of laneRoutes) {
      const old=this.state.lanes[r.service]
      this.state.lanes[r.service]={...old,limit:Math.min(old?.limit||this.capacity.initial[r.service],this.capacity.ceilings[r.service])}
    }
  }
  releaseRemovedAssignments() {
    for(const task of this.state.tasks) {
      if(task.stage!==0||!['pending','interrupted'].includes(task.status)||!task.baseService||baseRoutes.some(r=>r.service===task.baseService))continue
      task.reassignments=[...task.reassignments||[],{from:task.baseService,reason:'担当プールから除外されたため再割当',time:stamp()}]
      delete task.baseService;delete task.baseOptions
    }
  }
  async status(offset=0,limit=30) {await this.ready;return this.serial(()=>this.snapshot(offset,limit))}
  async run(parent) {
    await this.ready
    const result=await this.serial(async()=>{
      if(!this.state)throw Error('先に全体計画を import してください')
      if(this.parent && this.parent.session.header.id!==parent.session.header.id && this.active.size)throw Error('別のチャットが実行中です')
      this.parent=parent;this.state.owner=parent.session.header.id;this.state.enabled=true
      this.initLanes()
      this.releaseRemovedAssignments()
      for(const lane of Object.values(this.state.lanes))lane.disabled=false
      for(const t of this.state.tasks)if(t.status==='interrupted')t.status='pending'
      await this.save();return this.snapshot()
    })
    await this.pump();return {...result,...this.snapshot()}
  }
  async pump() {
    const selected=await this.serial(async()=>{
      const picked=[]
      let changed=false
      if(this.closed || !this.state?.enabled || !this.parent)return picked
      const byId=new Map(this.state.tasks.map(t=>[t.id,t]))
      const reviews=this.state.tasks.filter(t=>t.status==='review')
      for(const t of [...this.state.tasks].sort((a,b)=>(b.stage>0)-(a.stage>0))) {
        if(this.active.size>=this.capacity.total)break
        if(t.status!=='pending'||!t.dependsOn.every(id=>done(byId.get(id))))continue
        if(t.stage===0&&t.baseService&&this.state.lanes[t.baseService]?.disabled) {
          t.attempts.push({route:taskRoute(t).label,stopReason:'unavailable',startedAt:stamp(),endedAt:stamp(),
            error:'再開担当の利用枠切れを確認したため次の担当へ引き継ぎ'})
          this.advance(t,'Assigned subscription/service quota exhausted');changed=true
          if(t.status!=='pending')continue
        }
        if(t.stage===0&&baseRoutes.every(r=>this.state.lanes[r.service].disabled)) {
          t.stage=1
          changed=true
          t.attempts.push({route:'Cursor / Grok',stopReason:'unavailable',startedAt:stamp(),endedAt:stamp(),
            error:'通常プールの全サービスが利用枠を使い切ったため利用可能な次の担当へ引き継ぎ'})
        }
        if(t.stage===1&&this.state.lanes.codex.disabled) {
          t.attempts.push({route:routes[1].label,stopReason:'unavailable',startedAt:stamp(),endedAt:stamp(),
            error:'Astraのサブスク利用枠切れを確認済みのためOpusへ引き継ぎ'})
          this.advance(t,'Astra subscription quota exhausted');changed=true
          if(t.status!=='pending')continue
        }
        const running=[...this.active.keys()].map(id=>byId.get(id))
        if((t.stage>0&&running.some(x=>x.stage>0))||[...running,...reviews].some(x=>conflicts(t,x)))continue
        if(t.stage===0) {
          const candidates=baseRoutes.filter(r=>!t.baseService||t.baseService===r.service).map(r=>({r,
            count:running.filter(x=>taskRoute(x)?.service===r.service).length,lane:this.state.lanes[r.service]}))
            .filter(({count,lane})=>!lane.disabled&&count<lane.limit).sort((a,b)=>a.count-b.count)
          if(!candidates.length)continue
          t.baseService=candidates[0].r.service
        }
        const route=taskRoute(t)
        const lane=this.state.lanes[route.service]
        if(lane&&running.filter(x=>taskRoute(x)?.service===route.service).length>=lane.limit)continue
        t.status='running'
        t.attempts.push({route:route.label,model:route.model,startedAt:stamp()})
        this.active.set(t.id,{controller:new AbortController(),activation:null})
        picked.push(t.id)
      }
      if(picked.length||changed)await this.save()
      return picked
    })
    for(const id of selected) {
      const job=this.execute(id).catch(async error=>{
        this.onError(error)
        await this.serial(async()=>{this.state.enabled=false;this.state.error=error.message;await this.save()})
      })
      this.jobs.add(job);job.finally(()=>this.jobs.delete(job)).catch(this.onError)
    }
  }
  async execute(id) {
    const task=this.state.tasks.find(t=>t.id===id),entry=this.active.get(id),route=taskRoute(task)
    let result
    try {
      const activation=await this.start({parent:this.parent,task,route,signal:entry.controller.signal,outputSchema:resultSchema,
        prompt:workerPrompt(this.state,task,task.attempts.slice(0,-1))})
      entry.activation=activation
      await this.serial(async()=>{task.attempts.at(-1).childId=activation.childId;await this.save()})
      if(entry.controller.signal.aborted)await activation.dispose()
      result=await activation.result
    } catch(error) {result={stopReason:entry.controller.signal.aborted?'aborted':'error',diagnostic:error.message}}
    await this.serial(async()=>{
      const attempt=task.attempts.at(-1)
      attempt.endedAt=stamp();attempt.stopReason=result.stopReason
      if(result.output?.length)attempt.partialOutput=result.output.filter(b=>b.type==='text').map(b=>b.text).join('\n').slice(-16000)
      if(result.diagnostic)attempt.error=String(result.diagnostic).slice(0,4096)
      if(this.state.lanes[route.service] && !entry.controller.signal.aborted) {
        const lane=this.state.lanes[route.service],kind=pressure(attempt.error)
        if(kind) {
          lane.lastPressure={kind,message:attempt.error,time:stamp()}
          if(kind==='quota')lane.disabled=true
          else {lane.limit=Math.max(1,Math.floor(lane.limit/2));lane.cooldownSuccesses=lane.limit*2}
        } else if(result.stopReason==='completed') {
          lane.successes=(lane.successes||0)+1
          if(lane.cooldownSuccesses>0)lane.cooldownSuccesses--
          else if(this.capacity.adaptive)lane.limit=Math.min(lane.limit+1,this.capacity.ceilings[route.service])
        }
      }
      if(entry.controller.signal.aborted||result.stopReason==='aborted')task.status='interrupted'
      else if(result.stopReason==='refusal') {task.status='unresolved';attempt.error='Model declined this task; no automatic escalation.'}
      else if(/refus(?:al|ed)|safety policy|permission denied|not permitted|sandbox.{0,45}(?:failed|not available)/i.test(attempt.error||''))task.status='unresolved'
      else if(result.stopReason==='completed'&&result.structured) {
        const report=result.structured;attempt.report=report
        const outside=report.changedFiles.some(file=>{
          try {const p=taskPath(file);return !task.writes.some(w=>w==='*'||p===w||p.startsWith(w+'/'))}catch{return true}
        })
        if(outside){task.status='unresolved';attempt.error='宣言した変更範囲外のファイルが報告されました。親による確認が必要です。'}
        else if(report.status==='resolved'&&report.evidence.length&&report.summary.trim())task.status='review'
        else if(report.status==='external_blocked')task.status='unresolved'
        else this.advance(task,report.summary || '証拠を伴う解決報告がありません')
      } else this.advance(task,attempt.error || '構造化された完了報告がありません')
      this.active.delete(id);await this.save()
    })
    await this.pump()
  }
  advance(task,reason) {
    task.attempts.at(-1).handoff=reason
    let next=1
    // Initial Astra and previously failed fallback attempts must not run again
    // just because routing order changed. A manual retry remains explicit.
    while(next<routes.length&&(tried(task,routes[next])||this.state.lanes[routes[next].service]?.disabled))next++
    if(next<routes.length){task.stage=next;task.status='pending'}else task.status='unresolved'
  }
  async review(id,accepted,evidence,reason='') {
    await this.ready
    const result=await this.serial(async()=>{
      const t=this.state?.tasks.find(t=>t.id===id)
      if(t?.status!=='review')throw Error('確認待ちのタスクを指定してください')
      const proof=stringList(evidence,'review evidence',false)
      t.reviews=[...t.reviews||[],{accepted,evidence:proof,reason,time:stamp()}]
      if(accepted)t.status='accepted';else this.advance(t,reason || proof.join('\n'))
      await this.save();return this.snapshot()
    })
    await this.pump();return result
  }
  async pause() {
    await this.ready
    await this.serial(async()=>{if(this.state){this.state.enabled=false;await this.save()}})
    const stopped=[...this.active.values()].map(async e=>{e.controller.abort(Error('Task queue paused'));await e.activation?.dispose()})
    await Promise.allSettled(stopped);await Promise.allSettled([...this.jobs]);return this.status()
  }
  async retry(id,reason) {
    await this.ready
    return this.serial(async()=>{
      const t=this.state?.tasks.find(t=>t.id===id)
      if(t?.status!=='unresolved'||!reason?.trim())throw Error('未解決タスクと、新しい情報・再試行理由を指定してください')
      t.retryReason=reason;t.status='pending';await this.save();return this.snapshot()
    })
  }
  async close() {
    if(this.closed)return this.closing
    this.closed=true
    this.closing=(async()=>{try{await this.ready;await this.pause()}finally{this.lock?.stdin.end();await this.lockExit}})()
    return this.closing
  }
}
