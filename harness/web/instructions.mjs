import {createHash} from 'node:crypto'
import {constants} from 'node:fs'
import {access, lstat, mkdir, open, realpath} from 'node:fs/promises'
import {dirname, join, relative, resolve} from 'node:path'
import {resolveDshHome} from '@deepseek-ai/dsh-home-paths'
import {withFileLock, writeFileAtomic} from '@deepseek-ai/dsh-atomic-write'
import {FileSystemSkillProvider} from '@deepseek-ai/dsh-skill-filesystem'
import {isSkillName} from '@deepseek-ai/dsh-skill'
import {parse} from 'yaml'

const globalDocs = ['AGENTS.md']
const projectDocs = ['AGENTS.md','CLAUDE.md','AGENTS.local.md','CLAUDE.local.md']

async function skillProject(cwd) {
  for (let dir=cwd;;dir=dirname(dir)) {
    try {await access(join(dir,'.git')); return dir}
    catch (e) {if(e.code!=='ENOENT') throw e}
    if (dir===dirname(dir)) return cwd
  }
}

async function writableDocument(root, key) {
  if (!root.documents.includes(key) && !/^skills\/[a-zA-Z0-9_.-]+(?:\.md|\/SKILL\.md)$/.test(key)) throw Error('未対応のファイル名です')
  const skill=key.startsWith('skills/')
  const parts=(skill ? key.slice(7) : key).split('/')
  if (parts.some(part=>part==='.' || part==='..' || part.startsWith('.'))) throw Error('ファイル名が無効です')
  // Start from an authorized existing directory. Refuse linked components before creating children.
  const anchor = skill ? root.skillAnchor : root.base
  if (await realpath(anchor)!==resolve(anchor)) throw Error('保存先が変更されました')
  const path = skill ? join(root.skillsPath,...parts) : join(root.base,key)
  let parent = anchor
  for (const part of relative(anchor,dirname(path)).split('/').filter(Boolean)) {
    parent=join(parent,part)
    try {if (!(await lstat(parent)).isDirectory()) throw Error('リンク先には書き込めません')}
    catch(e) {if(e.code!=='ENOENT') throw e}
  }
  return path
}

async function readDoc(path) {
  let file
  try {
    file=await open(path,constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
    const stat=await file.stat()
    if (!stat.isFile() || stat.size>1024*1024) throw Error('1 MiB 以下の通常ファイルを選んでください')
    const body=await file.readFile('utf8')
    return {body,revision:createHash('sha256').update(body).digest('hex'),mode:stat.mode & 0o777}
  } catch(e) {
    if(e.code==='ENOENT') return {body:'',revision:null,mode:0o600}
    throw e
  } finally {await file?.close()}
}

function validateSkill(body) {
  const match=body.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)
  const data=match && parse(match[1])
  if (typeof data?.name!=='string' || !isSkillName(data.name) || typeof data.description!=='string' || !data.description.trim()) {
    throw Error('先頭の YAML に name（英小文字の kebab-case）と description が必要です')
  }
  for (const k of ['disable-model-invocation','user-invocable']) {
    if(k in data && !/^(?:true|false|yes|no|on|off|0|1)$/i.test(String(data[k]))) throw Error(k+' は真偽値で指定してください')
  }
  for(const k of ['disableModelInvocation','modelInvocable','userInvocable']) if (k in data) throw Error('未対応のスキル設定 '+k)
}

export function instructionsApi(ctx) {
  const controller=new AbortController()
  const filesystem=new FileSystemSkillProvider(ctx,{invalidate(){},signal:controller.signal},{watch:false})
  ctx.effect(()=>()=>{controller.abort();return filesystem.dispose()},'Studio skill catalog')
  return async request=>{
    const scopes=[{id:'global',title:'DSH 全体',path:resolveDshHome()},...ctx.workspaceRegistry.list().map(({id,title,path})=>({id,title,path}))]
    const query=new URL(request.url).searchParams
    const saving=request.method==='POST'
    if(saving && request.headers.get('content-type')?.split(';')[0]!=='application/json') throw Error('JSON が必要です')
    const args=saving ? await request.json() : Object.fromEntries(query)
    const scope=scopes.find(s=>s.id===(args.scope ?? 'global'))
    if(!scope) throw Error('未登録のワークスペースです')
    const global=scope.id==='global'
    const anchor=global ? scope.path : await skillProject(scope.path)
    const root={base:scope.path,skillAnchor:anchor,skillsPath:join(anchor,...(global?['skills']:['.dsh','skills'])),documents:global?globalDocs:projectDocs}
    if(typeof args.key==='string') {
      const path=await writableDocument(root,args.key)
      if (!saving) return {...await readDoc(path),path}
      if(typeof args.body!=='string' || Buffer.byteLength(args.body)>1024*1024) throw Error('文書は1 MiB 以下で入力してください')
      if(args.key.startsWith('skills/')) validateSkill(args.body)
      await mkdir(dirname(path),{recursive:true,mode:0o700})
      return withFileLock(path,async()=>{
        await writableDocument(root,args.key)
        const current=await readDoc(path)
        if(current.revision!==args.revision) throw Error('他の編集が保存されています。再読み込みして確認してください')
        await writeFileAtomic(path,args.body,{mode:current.mode})
        return {...await readDoc(path),path}
      })
    }
    if(saving) throw Error('保存するファイルが必要です')
    const options={cwd:global?undefined:scope.path,signal:request.signal}
    const [local,registered]=await Promise.all([filesystem.list(options),ctx.skills.list(options)])
    const all=Array.isArray(local)?local:local.candidates
    const skills=new Map(registered.map(s=>[s.name,s]))
    for(const s of [...all].sort((a,b)=>b.rank-a.rank)) skills.set(s.name,s)
    const rows=[...skills.values()].sort((a,b)=>a.name.localeCompare(b.name)).map(s=>{
      const rel=s.path && relative(root.skillsPath,s.path)
      const key=rel && /^(?!\.)[A-Za-z0-9_.-]+(?:\.md|\/SKILL\.md)$/.test(rel)?'skills/'+rel:null
      return {name:s.name,description:s.description,source:s.source,key}
    })
    return {scopes,documents:root.documents,skills:rows,skillsPath:root.skillsPath,complete:Array.isArray(local)||local.complete}
  }
}
