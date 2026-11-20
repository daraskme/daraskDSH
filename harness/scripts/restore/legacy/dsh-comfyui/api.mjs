import {randomInt,randomUUID} from 'node:crypto'
import {setTimeout as delay} from 'node:timers/promises'
import {comfyOrigin} from './proxy.mjs'

async function request(path, signal, options={}) {
  const timeout = AbortSignal.timeout(30000)
  const response = await fetch(new URL(path,comfyOrigin()),{
    ...options,signal:signal ? AbortSignal.any([signal,timeout]) : timeout,redirect:'error'
  })
  if (!response.ok) {
    const message = (await response.text()).slice(0,2200)
    throw Object.assign(Error('ComfyUI '+response.status+' '+message),{status:response.status})
  }
  return response.json()
}
const post = value => ({method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(value)})
export async function catalog({kind='checkpoints',query=''}, signal) {
  if (kind === 'node') return request('/object_info/'+encodeURIComponent(query),signal)
  if (kind === 'nodes') {
    const nodes = await request('/object_info',signal)
    const matches = Object.keys(nodes).filter(id=>id.toLowerCase().includes(query.toLowerCase()))
    return {total:matches.length,nodes:matches.slice(0,120)}
  }
  if (kind === 'folders') return request('/models',signal)
  return request('/models/'+encodeURIComponent(kind),signal)
}
export async function queue({graph,workflow}, signal) {
  if (!graph || Array.isArray(graph) || !Object.keys(graph).length ||
      !Object.values(graph).every(node=>typeof node?.class_type === 'string' && node.inputs && typeof node.inputs === 'object')) {
    throw Error('An API-format graph with class_type and inputs per node is required')
  }
  const result = await request('/prompt',signal,post({
    prompt:graph, client_id:randomUUID(), ...(workflow && {extra_data:{extra_pnginfo:{workflow}}})
  }))
  return {id:result.prompt_id,queueNumber:result.number}
}
export async function job({id,waitSeconds=0,includeGraph=false}, signal) {
  if (!/^[\da-f-]{36}$/i.test(id)) throw Error('Invalid ComfyUI prompt id')
  const deadline = Date.now()+1000*Math.min(60,Math.max(0,waitSeconds))
  for (;;) {
    const info = await request('/api/jobs/'+id,signal)
    if (!['pending','in_progress'].includes(info.status) || Date.now()>=deadline) {
      const files = []
      for (const [node,groups] of Object.entries(info.outputs || {})) {
        for (const [kind,items] of Object.entries(groups)) {
          if (!Array.isArray(items)) continue
          for (const item of items) {
            if (!item?.filename) continue
            const params = new URLSearchParams({filename:item.filename,subfolder:item.subfolder || '',type:item.type || 'output'})
            files.push({node,kind,filename:item.filename,url:'/comfy/view?'+params,localUrl:new URL('/view?'+params,comfyOrigin()).href})
          }
        }
      }
      return {id,status:info.status,files,...(info.execution_error && {error:info.execution_error.exception_message}),
        ...(includeGraph && {graph:info.workflow?.prompt,workflow:info.workflow?.extra_data?.extra_pnginfo?.workflow})}
    }
    await delay(1000,undefined,{signal})
  }
}
export async function workflows({action='list',name,workflow,overwrite=false},signal) {
  if (action === 'list') {
    try { return await request('/userdata?dir=workflows&recurse=true',signal) }
    catch (error) { if (error.status === 404) return []; throw error }
  }
  if (!name || !name.endsWith('.json') || /[\\\x00-\x1f]/.test(name) || name.split('/').some(p=>!p || p==='.' || p==='..')) throw Error('Use a relative .json workflow name')
  const path = '/userdata/'+encodeURIComponent('workflows/'+name)
  if (action === 'get') return request(path,signal)
  if (action !== 'save' || !Array.isArray(workflow?.nodes) || !Array.isArray(workflow?.links)) throw Error('Save requires an editor workflow with nodes and links')
  return request(path+'?overwrite='+overwrite,signal,post(workflow))
}

export async function generate({prompt,negative='',checkpoint,seed=randomInt(2**32),steps=20,width=512,height=512},signal) {
  if (!prompt.trim() || ![width,height].every(n=>Number.isInteger(n) && n>=64 && n<=2048 && n%8===0) ||
      !Number.isInteger(steps) || steps<1 || steps>100 || !Number.isSafeInteger(seed) || seed<0) throw Error('Supply a prompt, 64–2048 dimensions divisible by eight, 1–100 steps, and a nonnegative seed')
  const choices = await catalog({kind:'checkpoints'},signal)
  const selected = checkpoint || (choices.includes('v1-5-pruned-emaonly.safetensors') ? 'v1-5-pruned-emaonly.safetensors' : undefined)
  if (!choices.includes(selected)) throw Error('Choose an installed SD1.x or SDXL checkpoint: '+choices.join(', '))
  const graph = {
    '1':{class_type:'CheckpointLoaderSimple',inputs:{ckpt_name:selected}},
    '2':{class_type:'CLIPTextEncode',inputs:{clip:['1',1],text:prompt}},
    '3':{class_type:'CLIPTextEncode',inputs:{clip:['1',1],text:negative}},
    '4':{class_type:'EmptyLatentImage',inputs:{width,height,batch_size:1}},
    '5':{class_type:'KSampler',inputs:{model:['1',0],positive:['2',0],negative:['3',0],latent_image:['4',0],steps,seed,cfg:7,sampler_name:'euler',scheduler:'normal',denoise:1}},
    '6':{class_type:'VAEDecode',inputs:{samples:['5',0],vae:['1',2]}},
    '7':{class_type:'SaveImage',inputs:{images:['6',0],filename_prefix:'DSH/image'}}
  }
  const queued = await queue({graph},signal)
  return {...queued,seed,checkpoint:selected}
}
