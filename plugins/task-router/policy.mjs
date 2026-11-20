import {selectionId} from '../../harness/agents/options.mjs'

const route=r=>({...r,provider:'local-cli',model:selectionId(r.service,r.options)})
export const routingVersion=2
export const coordinatorRoute=route({service:'codex',options:{model:'gpt-6.2-astra',effort:'high',subscription:true},label:'Codex Astra 6.2'})
export const baseRoutes = Object.freeze([
  {service:'cursor',options:{},label:'Cursor'},
  {service:'grok',options:{},label:'Grok'}
].map(route))
export const routes = Object.freeze([
  {service:'cursor',options:{},label:'Cursor'},
  coordinatorRoute,
  {service:'claude',options:{model:'claude-opus-5-5',effort:'high'},label:'Claude Opus 5.5'}
].map(route))
// Historical reports keep their actual author even after a provider leaves the
// initial pool. Pending/interrupted assignments are released before execution.
export const historicalBaseRoutes=Object.freeze([coordinatorRoute])
export const taskRoute=task=>task.stage===0 ? [...baseRoutes,...historicalBaseRoutes].find(r=>r.service===task.baseService) : routes[task.stage]

export const resultSchema = {
  type:'object',additionalProperties:false,
  properties:{
    status:{type:'string',enum:['resolved','unresolved','external_blocked']},
    summary:{type:'string'},evidence:{type:'array',items:{type:'string'}},
    attempts:{type:'array',items:{type:'string'}},
    blockers:{type:'array',items:{type:'string'}},
    changedFiles:{type:'array',items:{type:'string'}}
  },required:['status','summary','evidence','attempts','blockers','changedFiles']
}

const idPattern = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/
const nonempty = v=>typeof v==='string' && v.trim().length>0
export function stringList(value,name,allowEmpty=true) {
  if(!Array.isArray(value) || (!allowEmpty && !value.length) || value.some(v=>!nonempty(v))) throw Error(name+' は文字列の配列で指定してください')
  return [...new Set(value)]
}
export function taskPath(value) {
  if(value==='*') return value
  if(!nonempty(value) || value.startsWith('/') || value.includes('\\') || /[*?[\]]/.test(value)) throw Error('変更・参照範囲は相対パスまたは * で指定してください')
  const path=value.replace(/\/+$/,'')
  if(path.split('/').some(p=>!p || p==='.' || p==='..')) throw Error('範囲のパスが不正です')
  return path
}
export function validatePlan(value) {
  if(!value || value.version!==1 || value.inventoryComplete!==true || !nonempty(value.objective)) throw Error('version:1、objective、inventoryComplete:true を含む全体計画が必要です')
  const sources=stringList(value.sources,'sources',false)
  if(!Array.isArray(value.phases) || !value.phases.length || !Array.isArray(value.tasks) || !value.tasks.length || value.tasks.length>20000) throw Error('phases と tasks（1〜20000件）が必要です')
  const phases=value.phases.map(p=>{
    if(!p || !idPattern.test(p.id) || !nonempty(p.title)) throw Error('フェーズの id / title が不正です')
    return {id:p.id,title:p.title}
  })
  if(new Set(phases.map(p=>p.id)).size!==phases.length) throw Error('フェーズIDが重複しています')
  const tasks=value.tasks.map(t=>{
    if(!t || !idPattern.test(t.id) || !nonempty(t.title) || !nonempty(t.prompt) || !phases.some(p=>p.id===t.phase)) throw Error('タスクの id / phase / title / prompt が不正です')
    return {id:t.id,phase:t.phase,title:t.title,prompt:t.prompt,
      dependsOn:stringList(t.dependsOn,'dependsOn'),acceptance:stringList(t.acceptance,'acceptance',false),
      reads:stringList(t.reads,'reads').map(taskPath),writes:stringList(t.writes,'writes').map(taskPath),
      baselineEvidence:stringList(t.baselineEvidence || [],'baselineEvidence')}
  })
  const byId=new Map(tasks.map(t=>[t.id,t]))
  if(byId.size!==tasks.length) throw Error('タスクIDが重複しています')
  const incoming=new Map(),following=new Map()
  for(const t of tasks) {
    incoming.set(t.id,t.dependsOn.length)
    for(const dep of t.dependsOn) {
      if(!byId.has(dep)) throw Error('存在しない依存タスク: '+dep)
      const next=following.get(dep)||[];next.push(t.id);following.set(dep,next)
    }
  }
  const ready=tasks.filter(t=>incoming.get(t.id)===0).map(t=>t.id)
  for(let at=0;at<ready.length;at++) for(const id of following.get(ready[at])||[]) {
    incoming.set(id,incoming.get(id)-1);if(incoming.get(id)===0)ready.push(id)
  }
  if(ready.length!==tasks.length) throw Error('タスクの依存関係が循環しています')
  return {version:1,objective:value.objective,sources,phases,tasks,inventoryComplete:true}
}
export function conflicts(a,b) {
  const overlap=(x,y)=>x==='*'||y==='*'||x===y||x.startsWith(y+'/')||y.startsWith(x+'/')
  return a.writes.some(x=>[...b.reads,...b.writes].some(y=>overlap(x,y))) || b.writes.some(x=>a.reads.some(y=>overlap(x,y)))
}
export function workerPrompt(plan,task,attempts) {
  return [
    'DSH_MANAGED_TASK: Work only on the task below. Use Harness tools. Do not delegate or start another queue. Do not commit/push or modify unrelated work.',
    'Read applicable AGENTS.md and current source-of-truth documents. Respect project execution/review gates. Do not rewrite frozen revisions. Codex Astra 6.2 with subscription authentication coordinates the project. Initial tasks are distributed only across Cursor and Grok; Astra is not an initial worker and Devin is excluded. Unresolved Cursor/Grok tasks advance to subscription Astra 6.2, then Claude Opus 5.5. The queue owns assignment and escalation. Legacy Cursor coordination, three-provider initial pools and Luna/Sol routing do not apply to this task.',
    'Use only your declared write scope; avoid touching shared ledgers or editing another worker\'s files. Stop at the acceptance criteria. Cite actual checks and artifact paths. Source-only/static findings are not runtime success or full-project completion.',
    'If you cannot resolve the technical problem after two materially different approaches, report unresolved with attempts/blockers. Missing authorization, unavailable external inputs, or project execution gates are external_blocked; do not bypass them. Do not repeatedly retry the same failing approach.',
    'Finish using structured_output. resolved requires concrete evidence for every acceptance criterion; the parent will independently review it. Include all changed relative file paths and outstanding limitations. Never claim tests ran when they did not.',
    JSON.stringify({objective:plan.objective,sources:plan.sources,task,previousAttempts:attempts.map(a=>({route:a.route,stopReason:a.stopReason,report:a.report,error:a.error,handoff:a.handoff,partialOutput:a.partialOutput,childId:a.childId}))})
  ].join('\n')
}
