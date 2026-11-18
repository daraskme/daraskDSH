import { readFile, mkdir, rename, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { randomUUID } from 'node:crypto'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { executable } from './cli.mjs'
import {devinModels} from './devin.mjs'
import {googleModels} from './google.mjs'

const run = promisify(execFile)
const levels = ['low','medium','high','xhigh']
export const capabilities = {
  auto: {models:[],efforts:[]},
  google: {models:[],efforts:[...levels,'max']},
  'devin-free': {models:[],efforts:[]}, devin:{models:[],efforts:[]},
  muse: {models:['muse-spark-1.3','muse-wonder-1.0','muse-ember-1.0'], efforts:['none','minimal',...levels,'max','ultra']},
  grok: {models:['grok-4.7','grok-4.7-build-fast','grok-4.6','grok-4.5'], efforts:levels},
  cursor: {models:['auto'], efforts:[]},
  codex: {models:[], efforts:[...levels,'max','ultra'], fast:true, subscription:true},
  claude: {models:['opus','sonnet','haiku','fable'], efforts:[...levels,'max'], fast:true, ultracode:true}
}
const dir = () => join(process.env.DSH_HOME || join(homedir(), '.local/state/darask-dsh'), 'cli-options')
function optionPath(name) {
  if (!Object.hasOwn(capabilities,name)) throw Error('Unknown CLI')
  return join(dir(), name + '.json')
}
export function validate(name, value) {
  const supported = capabilities[name]
  if (!supported || !value || typeof value !== 'object' || Array.isArray(value)) throw Error('CLI 設定が不正です')
  const options = {}
  if(name==='auto'&&Object.keys(value).length)throw Error('自動引き継ぎのモデルとエフォートは指定された順序に固定されています')
  for (const [key,v] of Object.entries(value)) {
    if (key === 'model' && typeof v === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_.,:/=[\]-]{0,180}$/.test(v)) options.model = v
    else if (key === 'effort' && supported.efforts.includes(v)) options.effort = v
    else if ((key === 'fast' || key === 'ultracode' || key === 'subscription') && supported[key] && typeof v === 'boolean') options[key] = v
    else throw Error(name + ' で利用できない設定: ' + key)
  }
  return options
}
export async function readOptions(name) {
  try { return validate(name,JSON.parse(await readFile(optionPath(name),'utf8'))) }
  catch (e) { if (e.code === 'ENOENT') return {}; throw e }
}
export async function saveOptions(name, values) {
  const path = optionPath(name), data = validate(name, values)
  await mkdir(dir(),{recursive:true,mode:0o700})
  const temp = path + '.' + randomUUID()
  await writeFile(temp,JSON.stringify(data),{mode:0o600})
  await rename(temp,path)
  return data
}
let catalog, expires = 0
export async function modelCatalog(refresh = false) {
  if (refresh) expires = 0
  if (catalog && Date.now() < expires) return catalog
  const result = structuredClone(capabilities)
  catalog = (async () => {
    await Promise.all([
      (async () => {
        result.google = await googleModels(await executable('google'))
      })().catch(()=>{result.google.unavailable=true;result.google.efforts=[]}),
      (async () => {
        const variants = await devinModels(await executable('devin'))
        const free = variants.filter(v=>v.free)
        result.devin = {...result.devin,models:variants.map(v=>v.id),variants}
        result['devin-free'] = {...result['devin-free'],models:free.map(v=>v.id),variants:free}
      })().catch(()=>{result.devin.unavailable=true;result['devin-free'].unavailable=true}),
      (async () => {
        const file = join(process.env.CODEX_HOME || join(homedir(),'.codex'),'models_cache.json')
        const {models} = JSON.parse(await readFile(file,'utf8'))
        const listed = models.filter(m => m.visibility === 'list')
        result.codex.models = listed.map(m => m.slug)
        result.codex.byModel = Object.fromEntries(listed.map(m => [m.slug,{efforts:m.supported_reasoning_levels?.map(e => e.effort) || [],fast:m.service_tiers?.some(t=>t.id==='priority') || false}]))
      })().catch(()=>{}),
      (async () => {
        const {stdout} = await run(await executable('cursor'),['--list-models'],{timeout:18000})
        const models = [...stdout.matchAll(/^([\w.:[\]=,-]+) - /gm)].map(m=>m[1])
        if (models.length) result.cursor.models = models
      })().catch(()=>{})
    ])
    return result
  })()
  expires = Date.now() + 600000
  return catalog
}
export async function cliOptions(name) {
  const settings = await readOptions(name)
  if (process.env.DARASK_DSH_MODEL !== name) return settings
  const overrides = {}
  for (const key of ['model','effort','fast','ultracode']) {
    const v = process.env['DARASK_OPT_' + key.toUpperCase()]
    if (v !== undefined) overrides[key] = ['fast','ultracode'].includes(key) ? v === '1' : v
  }
  return validate(name,{...settings,...overrides})
}
export function optionArgs(name, options) {
  const {model,effort,fast,ultracode,subscription} = options
  // Antigravity embeds effort in model slugs and rejects conflicting flags.
  const cliModel = name === 'google' && effort && model ? model.replace(/-(low|medium|high|xhigh|max)$/,'-'+effort) : model
  const args = cliModel ? ['--model',cliModel] : []
  if (effort) {
    if (name === 'codex') args.push('-c','model_reasoning_effort=' + JSON.stringify(effort))
    else args.push(name === 'claude' || name === 'google' ? '--effort' : '--reasoning-effort', effort)
  }
  if (name === 'codex' && fast) args.push('-c','service_tier="priority"')
  if (name === 'codex' && subscription) args.push('-c','forced_login_method="chatgpt"',
    '-c','model_provider="openai"','-c','cli_auth_credentials_store="file"')
  if (name === 'claude' && (fast || ultracode)) args.push('--settings', JSON.stringify({...fast && {fastMode:true}, ...ultracode && {ultracode:true}}))
  return args
}

// Explicit selections are complete snapshots, so another chat or a later change
// to service defaults cannot change an in-progress conversation's model/options.
export function selectionId(service, options) {
  return service + '@' + Buffer.from(JSON.stringify(validate(service,options))).toString('base64url')
}
export function parseSelection(id) {
  if (typeof id !== 'string' || id.length > 2048) throw Error('モデル選択が不正です')
  const at = id.indexOf('@'), service = at < 0 ? id : id.slice(0,at)
  if (!Object.hasOwn(capabilities,service)) throw Error('Unknown CLI model: '+service)
  if (at < 0) return {service}
  const encoded = id.slice(at+1)
  if (!/^[A-Za-z0-9_-]+$/.test(encoded)) throw Error('モデル選択が不正です')
  return {service,options:validate(service,JSON.parse(Buffer.from(encoded,'base64url').toString('utf8')))}
}
