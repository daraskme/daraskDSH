import {realpathSync, statSync} from 'node:fs'
import {realpath, stat} from 'node:fs/promises'
import {isAbsolute} from 'node:path'
import {appendPluginRecord, pluginRecordOf} from '@deepseek-ai/dsh-session'
import {z} from 'zod'

export const name = 'darask-workspace-folders'
export const inject = ['sessions', 'sessionProjections', 'sandboxPolicy']

const type = 'plugin:darask/workspace-folders'
const folderList = z.array(z.string().min(1).refine(isAbsolute))
export const projection = {
  key:'daraskWorkspaceFolders', stateVersion:1,
  stateSchema:z.object({owner:z.string(), folders:folderList, revision:z.number().int()}),
  init:header => ({owner:header.id, folders:[], revision:0}),
  apply(state, event) {
    if (event.type !== type) return state
    const record = pluginRecordOf(event)
    if (record?.data?.owner !== state.owner) return state
    const folders = folderList.parse(record.data.folders)
    return {...state, folders, revision:event.seq+1}
  },
}

export async function canonicalFolders(paths, primary) {
  if (!Array.isArray(paths)) throw Error('フォルダの一覧が必要です')
  const main = await realpath(primary)
  const folders = new Set()
  for (const path of paths) {
    if (typeof path !== 'string' || !isAbsolute(path)) throw Error('フォルダは絶対パスで指定してください')
    const canonical = await realpath(path)
    if (!(await stat(canonical)).isDirectory()) throw Error('フォルダを選んでください: '+path)
    if (canonical !== main) folders.add(canonical)
  }
  return [...folders]
}

export function policyWithFolders(policy, folders) {
  if (policy.mode !== 'workspace-write' || !folders.length) return policy
  // A directory picked earlier may have been removed or replaced by a symlink.
  const current = folders.filter(path => {
    try { return realpathSync(path) === path && statSync(path).isDirectory() }
    catch { return false }
  })
  return {...policy, additionalWritableRoots:[...new Set([...(policy.additionalWritableRoots ?? []), ...current])]}
}

export async function apply(ctx) {
  const handle = await ctx.sessionProjections.register(projection)
  // DSH exposes a single resolver used by fs, bash and terminals. Preserve the
  // original receiver (Cordis scopes services) and undo the decoration on unload.
  const policy = ctx.sandboxPolicy
  const descriptor = Object.getOwnPropertyDescriptor(policy, 'resolve')
  const resolve = descriptor?.value ?? Object.getPrototypeOf(policy).resolve
  ctx.effect(() => {
    Object.defineProperty(policy, 'resolve', {configurable:true, writable:true, value:function(request) {
      const base = resolve.call(this, request)
      return request?.session ? policyWithFolders(base, handle.stateOf(request.session).folders) : base
    }})
    return () => {
      if (descriptor) Object.defineProperty(policy,'resolve',descriptor)
      else delete policy.resolve
    }
  }, 'Session workspace folders in confinement policy')

  ctx.on('system-prompt/assemble', async (_assembly, context, next) => {
    const assembly = await next()
    const session = context.agent?.session
    const folders = session && handle.stateOf(session).folders
    if (folders?.length) assembly.contexts.push({
      name:'darask:workspace-folders', interpolate:false,
      text:'Additional workspace folders selected for this chat (use their absolute paths): '+JSON.stringify(folders),
    })
    return assembly
  })

  ctx.inject(['connection','workspaceRegistry'], web => {
    web.effect(() => web.connection.fetch.register({
      path:'/api/darask.workspaceFolders', methods:['GET','POST'], requestBody:'buffered',
      async fetch(request) {
        const reply = (data,status=200) => Response.json(data,{status,headers:{'cache-control':'no-store'}})
        const id = new URL(request.url).searchParams.get('session')
        const session = ctx.sessions.get(id)
        if (!session) return reply({error:'このチャットを開き直してください。'},404)
        const primary = session.header.cwd ?? policy.workspaceRoot
        try {
          if (request.method === 'POST') {
            if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') return reply({error:'JSON が必要です'},415)
            const body = await request.json()
            const folders = await canonicalFolders(body.folders,primary)
            if (ctx.sessions.get(id) !== session) return reply({error:'チャットを開き直してください。'},409)
            const current = handle.stateOf(session)
            if (current.revision !== body.revision) return reply({error:'別の画面でフォルダが変更されました。開き直して確認してください。'},409)
            if (JSON.stringify(folders) !== JSON.stringify(current.folders)) appendPluginRecord(session,type,{owner:id,folders})
          }
          const {folders,revision} = handle.stateOf(session)
          return reply({primary, folders, revision, workspaces:web.workspaceRegistry.list().map(ws => ({path:ws.path,title:ws.title}))})
        } catch (error) { return reply({error:error.message},400) }
      },
    }), 'Manage folders for the open session')
  })
}
