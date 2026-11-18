import {execFile} from 'node:child_process'
import {promisify} from 'node:util'
const run = promisify(execFile)

// agy models reports account-specific, tab-separated IDs and labels.
export function parseGoogleModels(stdout) {
  const groups = new Map()
  for (const line of stdout.split('\n')) {
    const match = /^([\w.-]+)\t(.+)$/.exec(line.trim())
    if (!match) continue
    const [,id,label] = match
    const level = /-(low|medium|high|xhigh|max)$/.exec(id)?.[1]
    const key = level ? id.slice(0,-level.length-1) : id
    const rows = groups.get(key) || []
    rows.push({id,label,level}); groups.set(key,rows)
  }
  if (!groups.size) throw Error('Google のモデル一覧を取得できません。dsh login google でログインしてください。')
  const variants = [], byModel = {}
  for (const rows of groups.values()) {
    const first = rows.find(r=>r.level==='medium') || rows[0]
    const efforts = rows.length>1 ? ['low','medium','high','xhigh','max'].filter(e=>rows.some(r=>r.level===e)) : []
    const label = efforts.length ? first.label.replace(/ \((Low|Medium|High|XHigh|Max)\)$/i,'') : first.label
    variants.push({id:first.id,label})
    for (const row of rows) byModel[row.id] = {efforts}
  }
  return {models:variants.map(v=>v.id),variants,byModel,efforts:[],authenticated:true}
}

export async function googleModels(file) {
  const {stdout} = await run(file,['models'],{timeout:18000,maxBuffer:1024*1024,env:googleEnvironment()})
  return parseGoogleModels(stdout)
}

// This service uses the personal Google subscription. An inherited API key or
// ADC variable must not silently switch it to a separately billed Cloud project.
export function googleEnvironment(env = process.env) {
  return Object.fromEntries(Object.entries(env).filter(([key])=>
    /^(HOME|USER|LOGNAME|PATH|SHELL|LANG|LC_.*|TMPDIR|TERM|XDG_.*|DBUS_SESSION_BUS_ADDRESS|https?_proxy|all_proxy|no_proxy|SSL_CERT_FILE)$/i.test(key)))
}
