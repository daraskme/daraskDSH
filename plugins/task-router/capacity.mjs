import {availableParallelism,freemem} from 'node:os'

// CLI account-wide concurrency ceilings are not published. This is a local
// resource budget, not a vendor/account limit. Two I/O-bound workers per CPU,
// with 2 GiB available RAM per worker, and a configurable operational ceiling.
export function capacityConfig(env=process.env,resources={cpus:availableParallelism(),memory:freemem()}) {
  const integer=(name,fallback)=>{
    const n=env[name]===undefined?fallback:Number(env[name])
    if(!Number.isInteger(n)||n<1||n>128)throw Error(name+' は1〜128の整数で指定してください')
    return n
  }
  const total=integer('DSH_TASK_MAX_PARALLEL',Math.max(1,Math.min(128,resources.cpus*2,Math.floor(resources.memory/2**31))))
  return {adaptive:true,total,ceilings:{cursor:Math.min(total,integer('DSH_TASK_CURSOR_MAX',total)),
    grok:Math.min(total,integer('DSH_TASK_GROK_MAX',total)),
    codex:1},initial:{cursor:4,grok:4,codex:1}}
}
export function pressure(error='') {
  if(/quota.{0,35}(?:exhaust|exceed)|usage limit|credit.{0,20}(?:exhaust|insufficient)|billing limit/i.test(error))return 'quota'
  if(/429|too many (?:requests|concurrent)|concurren(?:cy|t).{0,30}limit|rate.?limit|resource[_ -]?exhausted/i.test(error))return 'concurrency'
  return null
}
