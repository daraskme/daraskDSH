import {readFile,mkdir,writeFile,rename} from 'node:fs/promises'
import {dirname,join} from 'node:path'
import {homedir} from 'node:os'
import {randomUUID} from 'node:crypto'
import {routes,routingVersion} from '../../plugins/task-router/policy.mjs'

function fileFor(sessionId) {
  if(!/^[A-Za-z0-9_-]{1,180}$/.test(sessionId || ''))throw Error('自動引き継ぎには有効なセッションIDが必要です')
  return join(process.env.DSH_HOME || join(homedir(),'.local/state/darask-dsh'),'routing',sessionId+'.json')
}
export async function routingState(sessionId) {
  if(!sessionId)return {stage:0,history:[],routingVersion}
  try {
    const state=JSON.parse(await readFile(fileFor(sessionId),'utf8'))
    if(!Number.isInteger(state.stage)||state.stage<0||state.stage>=routes.length||!Array.isArray(state.history))throw Error('引き継ぎ記録が不正です')
    if(state.routingVersion===undefined) {
      // Resume the actual old model, without re-labelling its saved history.
      if(state.stage>0)state.stage=state.stage===1?2:1
      state.routingVersion=routingVersion
    } else if(state.routingVersion!==routingVersion)throw Error('未対応の引き継ぎ構成です')
    return state
  }catch(e){if(e.code==='ENOENT')return {stage:0,history:[],routingVersion};throw e}
}
async function persist(sessionId,state) {
  if(!sessionId)return
  const file=fileFor(sessionId);await mkdir(dirname(file),{recursive:true,mode:0o700})
  const temp=file+'.'+randomUUID();await writeFile(temp,JSON.stringify(state,null,2)+'\n',{mode:0o600});await rename(temp,file)
}
export async function advanceRouting(sessionId,reason,evidence=[]) {
  const state=await routingState(sessionId)
  state.history.push({route:routes[state.stage].label,reason,evidence,time:new Date().toISOString()})
  const next=routes.findIndex((route,i)=>i>0&&!state.history.some(item=>item.route===route.label))
  if(next<0) {
    state.exhausted=true;await persist(sessionId,state)
    return {...state,route:routes[state.stage].label}
  }
  state.stage=next;state.exhausted=false;await persist(sessionId,state)
  return {...state,route:routes[state.stage].label}
}
export async function runAuto(options,invoke) {
  let state=await routingState(options.sessionId)
  if(state.exhausted)throw Error('引き継ぎ先でも未解決です。記録を確認し、新しい情報を加えて別のモデルを選択してください。')
  while(true) {
    options.signal?.throwIfAborted()
    const route=routes[state.stage]
    try {return await invoke(route,state)}
    catch(error) {
      options.signal?.throwIfAborted()
      if(options.purpose)throw error
      if(/refus(?:al|ed)|safety policy|permission denied|not permitted/i.test(error.message))throw error
      const reason=error.message.slice(-4096)
      if(options.sessionId)state=await advanceRouting(options.sessionId,reason)
      else if(state.stage<routes.length-1)state={...state,stage:state.stage+1,history:[...state.history,{route:route.label,reason}]}
      else state={...state,exhausted:true}
      if(state.exhausted)throw new Error('Cursor → Astra 6.2 → Opus 5.5 の全段階で継続できませんでした。最後のエラー: '+reason,{cause:error})
    }
  }
}
