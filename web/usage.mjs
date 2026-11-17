import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { createInterface } from 'node:readline'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { executable } from '../agents/cli.mjs'

const run = promisify(execFile)
const home = homedir()
function windowOf(label, used, reset) {
  if (typeof used !== 'number' || !Number.isFinite(used)) return null
  return {label, remaining:Math.min(100,Math.max(0,100-used)), reset:reset ? new Date(reset).toISOString() : null}
}
async function getJson(url, init) {
  const response = await fetch(url,{...init,signal:AbortSignal.timeout(10000)})
  if (!response.ok) throw Error('HTTP ' + response.status)
  return response.json()
}
async function codexUsage() {
  const proc = spawn(await executable('codex'),['app-server'],{stdio:['pipe','pipe','ignore']})
  const lines = createInterface({input:proc.stdout})
  function send(value) { proc.stdin.write(JSON.stringify(value)+'\n') }
  proc.stdin.on('error',()=>{})
  const timeout = setTimeout(()=>proc.kill(),12000)
  try {
    send({id:1,method:'initialize',params:{clientInfo:{name:'darask_studio',version:'1'}}})
    for await (const line of lines) {
      const msg = JSON.parse(line)
      if (msg.error) throw Error('Codex usage unavailable')
      if (msg.id === 1) {
        send({method:'initialized'})
        send({id:2,method:'account/rateLimits/read'})
      } else if (msg.id === 2) {
        const limits = msg.result.rateLimits
        return ['primary','secondary'].map(key=>{
          const w = limits?.[key]
          return w && windowOf(w.windowDurationMins ? w.windowDurationMins/60+'h' : key,w.usedPercent,w.resetsAt*1000)
        }).filter(Boolean)
      }
    }
    throw Error('Codex app-server ended')
  } finally { clearTimeout(timeout); lines.close(); proc.stdin.end(); proc.kill() }
}
async function terminalUsage(name) {
  const {stdout} = await run(fileURLToPath(new URL('../.venv/bin/python',import.meta.url)),
    [fileURLToPath(new URL('./terminal_usage.py',import.meta.url)),name,await executable(name)],{timeout:16000})
  return JSON.parse(stdout)
}
const readers = {
  async muse() {
    const {stdout} = await run(await executable('muse'),['usage','--json'],{timeout:13000})
    const usage = JSON.parse(stdout).allowance
    return ['window','weekly'].map(key=>usage[key] && windowOf(key,usage[key].usedPercent,usage[key].resetsAtMs)).filter(Boolean)
  },
  grok:()=>terminalUsage('grok'),
  devin:()=>terminalUsage('devin'),
  async cursor() {
    const auth = JSON.parse(await readFile(join(home,'.config/cursor/auth.json'),'utf8'))
    const data = await getJson('https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage',{
      method:'POST',headers:{Authorization:'Bearer '+auth.accessToken,'Content-Type':'application/json','Connect-Protocol-Version':'1'},body:'{}'})
    return [windowOf('billing cycle',data.planUsage?.totalPercentUsed,Number(data.billingCycleEnd))].filter(Boolean)
  },
  codex:codexUsage,
  async claude() {
    const path = join(process.env.CLAUDE_CONFIG_DIR || join(home,'.claude'),'.credentials.json')
    const token = process.env.CLAUDE_CODE_OAUTH_TOKEN || JSON.parse(await readFile(path,'utf8')).claudeAiOauth.accessToken
    const data = await getJson('https://api.anthropic.com/api/oauth/usage',{headers:{Authorization:'Bearer '+token,'anthropic-beta':'oauth-2025-04-20'}})
    return ['five_hour','seven_day'].map(key=>data[key] && windowOf(key,data[key].utilization,data[key].resets_at)).filter(Boolean)
  }
}
const cache = new Map()
export async function usage() {
  return Object.fromEntries(await Promise.all(Object.keys(readers).map(async name=>{
    let record = cache.get(name)
    if (!record || Date.now() >= record.until) {
      const previous = record?.value
      record = {until:Date.now()+180000,value:previous}
      record.pending = readers[name]().then(windows=>{
        if (!windows.length) throw Error('no windows')
        return record.value = {windows,updatedAt:new Date().toISOString()}
      }).catch(()=>({...(previous || {windows:[]}),unavailable:true}))
      cache.set(name,record)
    }
    return [name,await record.pending]
  })))
}
