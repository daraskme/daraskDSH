import {execFile} from 'node:child_process'
import {promisify} from 'node:util'
const run=promisify(execFile)

export async function devinModels(file,signal) {
  const {stdout}=await run(file,['models','list','--format','json'],{signal,timeout:15000,maxBuffer:5*1024*1024})
  const {families}=JSON.parse(stdout)
  return families.flatMap(family=>family.variants.map(variant=>({
    id:variant.model_uid,label:variant.label,
    free:variant.cost_tier==='Free' && (!variant.cost_summary || /^free$/i.test(variant.cost_summary)),
    price:variant.cost_summary || (variant.cost_tier==='Free' ? '無料' : variant.cost_tier) || '料金未確認'
  })))
}
export async function selectDevin(file,route,options,signal) {
  if(route==='devin' && options.model) return options
  const free=(await devinModels(file,signal)).filter(m=>m.free)
  const selected=options.model ? free.find(m=>m.id===options.model) : free.find(m=>m.id==='swe-2-high') || free[0]
  if(!selected) throw Error('このアカウントで無料と確認できるモデルを選んでください')
  return {...options,model:selected.id}
}
