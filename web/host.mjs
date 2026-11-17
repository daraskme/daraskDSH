import { readOptions, saveOptions, modelCatalog, capabilities } from '../agents/options.mjs'
import { usage } from './usage.mjs'
import { pickNativeDirectory } from '@deepseek-ai/dsh-host-directory-picker-native'
export const inject = ['connection']
export function apply(ctx) {
  function route(path,methods,fn) {
    ctx.effect(()=>ctx.connection.fetch.register({path,methods,requestBody:'buffered',fetch:async request=>{
      try { return Response.json(await fn(request),{headers:{'cache-control':'no-store'}}) }
      catch (error) { return Response.json({error:error.message},{status:400}) }
    }}),'darask studio '+path)
  }
  route('/api/darask.usage',['GET'],usage)
  let desktopPickerOpen = false
  route('/api/darask.pickWorkspace',['POST'],async request=>{
    if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') throw Error('JSON が必要です')
    if (process.platform !== 'linux' || !(process.env.DISPLAY || process.env.WAYLAND_DISPLAY)) {
      throw Error('Ubuntu デスクトップに接続できません')
    }
    if (desktopPickerOpen) throw Error('Ubuntu の選択画面は既に開いています')
    desktopPickerOpen = true
    try {
      return {path:await pickNativeDirectory(request.signal)}
    } finally {
      desktopPickerOpen = false
    }
  })
  route('/api/darask.options',['GET','POST'],async request=>{
    if (request.method === 'POST') {
      if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') throw Error('JSON が必要です')
      const {agent,values} = await request.json()
      return saveOptions(agent,values)
    }
    return {catalog:await modelCatalog(),values:Object.fromEntries(await Promise.all(Object.keys(capabilities).map(async name=>[name,await readOptions(name)])))}
  })
}
