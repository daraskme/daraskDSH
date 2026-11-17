import {defineTool} from '@deepseek-ai/dsh-tools'
import {catalog,generate,job,queue,workflows} from './api.mjs'
import {installEditor} from './proxy.mjs'

export const name = 'darask-dsh-comfyui'
export const inject = ['tools']
const string = {type:'string'}
const integer = {type:'integer'}
const object = {type:'object',additionalProperties:true}

export function apply(ctx) {
  ctx.inject(['webServer','connection'],installEditor)
  function tool(name,description,parameters,execute,readOnly=false) {
    ctx.tools.register(defineTool({name,description,parameters,
      output:{schema:{type:'string'},render:(_,value)=>[{type:'text',text:value}]},
      timeoutMs:95000,isConcurrencySafe:()=>readOnly,
      execute:async(args,exec)=>JSON.stringify(await execute(args,exec.signal))
    }))
  }
  tool('comfyui_catalog','Inspect installed ComfyUI models and nodes. kind=folders lists model categories, checkpoints or loras lists files, nodes searches node names, node returns the schema of an exact class in query.',
    {kind:string,query:string},catalog,true)
  tool('comfyui_generate','Generate an image on the local GPU from a text prompt using the installed standard SD1.x/SDXL checkpoint. Returns id; call comfyui_job with waitSeconds=60 and show the resulting file link.',
    {prompt:{...string,required:true},negative:string,checkpoint:string,seed:integer,width:integer,height:integer,steps:integer},generate)
  tool('comfyui_run','Queue a custom ComfyUI API-format node graph keyed by node ID. Each node has class_type and inputs. Optionally attach the corresponding editor workflow as PNG metadata. Retrieve outputs with comfyui_job.',
    {graph:{...object,required:true},workflow:object},queue)
  tool('comfyui_job','Read or wait for a specific job. Outputs contain Studio-relative URLs to share with the user. includeGraph also retrieves the executed graph, including jobs run from the editor.',
    {id:{...string,required:true},waitSeconds:integer,includeGraph:{type:'boolean'}},job,true)
  tool('comfyui_workflow','List, read or save named ComfyUI editor workflows shared with the user. Use get before modifying a workflow. save accepts editor-format nodes and links; overwrite=true replaces an existing named workflow.',
    {action:{...string,enum:['list','get','save']},name:string,workflow:object,overwrite:{type:'boolean'}},workflows)
}
