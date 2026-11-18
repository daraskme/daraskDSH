import {defineTool} from '@deepseek-ai/dsh-tools'
import {readFile,realpath} from 'node:fs/promises'
import {dirname,join,resolve} from 'node:path'
import {homedir} from 'node:os'
import {randomUUID} from 'node:crypto'
import {TaskQueue} from './queue.mjs'
import {advanceRouting,routingState} from '../../agents/auto.mjs'
import {cliOptions,selectionId} from '../../agents/options.mjs'
import {capacityConfig} from './capacity.mjs'
import {coordinatorRoute} from './policy.mjs'

export const name='darask-task-router'
export const inject=['tools','subagents','systemPrompt','workingDirectory']
const instructions=await readFile(new URL('TWA_INSTRUCTIONS.md',import.meta.url),'utf8')
const twaRoot=resolve(process.env.DSH_TWA_ROOT || '/mnt/6tb/TWA/totalwararena-godot')
const inTwa=cwd=>cwd===dirname(twaRoot)||cwd===twaRoot||cwd.startsWith(twaRoot+'/')

export function apply(ctx) {
  const queues=new Map(),children=new Set(),signals=new WeakMap()
  const stateRoot=join(process.env.DSH_HOME || join(homedir(),'.local/state/darask-dsh'),'task-runs')
  async function getQueue(agent) {
    if(!agent || agent.session.header.parentSession)throw Error('タスクキューの操作は親チャットが行ってください')
    const cwd=ctx.workingDirectory.get(agent.session),root=await realpath(inTwa(cwd)?twaRoot:cwd)
    const capacity=capacityConfig()
    capacity.total=Math.min(capacity.total,ctx.subagents.config.maxActiveSubagents.get())
    if(!queues.has(root))queues.set(root,new TaskQueue({root,stateRoot,capacity,onError:e=>ctx.logger.warn('task-router: '+e.message),
      start:async spec=>{
        if(spec.task.stage===0) {
          spec.task.baseOptions=spec.route.service==='codex' ? {...coordinatorRoute.options} : spec.task.baseOptions ?? await cliOptions(spec.route.service)
          spec.task.attempts.at(-1).model=selectionId(spec.route.service,spec.task.baseOptions)
        }
        const childId='session-'+randomUUID()
        children.add(childId)
        try {
          const activation=await ctx.subagents.startActivation({provider:'spawn',childId,
            label:spec.task.id+' · '+spec.route.label,delivery:'parent',signal:spec.signal,
            request:{parent:spec.parent,cwd:root,maxDepth:ctx.subagents.resolveMaxDepth(),
              agentOptions:{provider:spec.route.provider,model:spec.task.attempts.at(-1).model},outputSchema:spec.outputSchema,
              prompt:[{type:'text',text:spec.prompt}]}})
          activation.result.finally(()=>children.delete(childId)).catch(()=>{})
          return activation
        } catch(e){children.delete(childId);throw e}
      }}))
    return queues.get(root)
  }
  ctx.systemPrompt.context({name:'darask:twa-routing',order:650,required:true,interpolate:false,text:({agent})=>
    agent && inTwa(ctx.workingDirectory.get(agent.session)) ? instructions : ''})
  ctx.on('agent/created',({agent})=>{
    const current=agent.model.current()
    if(inTwa(ctx.workingDirectory.get(agent.session)) && current?.provider==='local-cli' &&
      (!agent.session.header.parentSession || current.model.split('@')[0]==='codex')) {
      agent.model.select({provider:coordinatorRoute.provider,model:coordinatorRoute.model})
    }
  })
  ctx.tools.guard(exec=>{
    if(children.has(exec.agent?.session.header.id)&&['subagent','task_queue'].includes(exec.name))
      return 'この作業は管理キューの子タスクです。再委任せず、未解決なら structured_output で報告してください。'
    if(exec.agent&&inTwa(ctx.workingDirectory.get(exec.agent.session))&&exec.name==='subagent'&&
      ['devin','devin-free'].includes(String(exec.arguments?.model||'').split('@')[0]))
      return 'ユーザーの現在の指定ではDevinは担当から除外されています。統括はサブスク枠Astra 6.2、通常担当はCursor・Grokのみ、未解決分はAstra 6.2→Opus 5.5です。'
  })
  ctx.on('agent/pre-step',async({agent,signal},next)=>{
    if(!agent.session.header.parentSession && signals.get(agent)!==signal) {
      signals.set(agent,signal)
      signal.addEventListener('abort',()=>{
        for(const q of queues.values())if(q.parent===agent)q.pause().catch(e=>ctx.logger.warn(e.message))
      },{once:true})
    }
    return next()
  })
  ctx.on('agent/disposed',({agent})=>{
    for(const q of queues.values())if(q.parent===agent)q.pause().catch(e=>ctx.logger.warn(e.message))
  })
  ctx.tools.register(defineTool({name:'task_queue',
    description:'Manage a persistent task plan. Subscription Codex Astra 6.2 coordinates TWA; Devin is excluded. import validates the complete phase/task JSON at planPath. run starts/resumes independent tasks only across Cursor and Grok. Astra is not an initial worker. Adaptive maximum mode grows concurrency toward the local CPU/RAM budget and reduces it on service limits; see maxParallel and basePool in status for actual limits. Unresolved Cursor/Grok work advances to subscription Astra 6.2, then Claude Opus 5.5. Only one fallback worker runs at a time. status returns counts and paged reports with review/unresolved tasks first. review accepts independently verified evidence or rejects to escalate. pause stops the queue. retry reopens an unresolved task at its current stage when new information is available; call run after retry. escalate advances only general-purpose auto-service parents; TWA keeps its Astra coordinator. Never use escalation to bypass permissions or a refusal. Do not poll while children are running: their native completion notices wake this chat. Status/UNRESOLVED.md are authoritative.',
    parameters:{action:{type:'string',enum:['import','run','status','review','pause','retry','escalate'],required:true},
      planPath:{type:'string'},taskId:{type:'string'},accepted:{type:'boolean'},evidence:{type:'array',items:{type:'string'}},reason:{type:'string'},
      offset:{type:'integer'},limit:{type:'integer'}},
    output:{schema:{type:'string'},render:(_,value)=>[{type:'text',text:value}]},
    timeoutMs:60000,isConcurrencySafe:()=>false,
    execute:async(args,exec)=>{
      if(args.offset!==undefined&&(!Number.isInteger(args.offset)||args.offset<0))throw Error('offset は0以上の整数です')
      if(args.limit!==undefined&&(!Number.isInteger(args.limit)||args.limit<1||args.limit>100))throw Error('limit は1〜100です')
      const q=await getQueue(exec.agent);let result
      if(args.action==='import'){if(!args.planPath)throw Error('planPath が必要です');result=await q.importPlan(args.planPath)}
      if(args.action==='run'){signals.get(exec.agent)?.throwIfAborted();result=await q.run(exec.agent)}
      if(args.action==='status')result=await q.status(args.offset||0,args.limit||30)
      if(args.action==='pause')result=await q.pause()
      if(args.action==='review') {
        if(typeof args.accepted!=='boolean')throw Error('accepted を指定してください')
        result=await q.review(args.taskId,args.accepted,args.evidence,args.reason)
      }
      if(args.action==='retry')result=await q.retry(args.taskId,args.reason)
      if(args.action==='escalate') {
        if(exec.agent.model.current()?.model.split('@')[0]!=='auto')throw Error('親の引き継ぎは「自動引き継ぎ」サービスで利用してください')
        if(!args.reason?.trim()||!args.evidence?.length)throw Error('未解決の理由と試行の証拠を指定してください')
        result=await advanceRouting(exec.agent.session.header.id,args.reason,args.evidence)
      }
      return JSON.stringify({...result,coordinator:inTwa(ctx.workingDirectory.get(exec.agent.session))
        ? {fixed:true,provider:coordinatorRoute.provider,model:coordinatorRoute.model,label:coordinatorRoute.label,subscription:true}
        : await routingState(exec.agent.session.header.id)})
    }
  }))
  ctx.effect(()=>async()=>{await Promise.allSettled([...queues.values()].map(q=>q.close()))})
}
