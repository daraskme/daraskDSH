import assert from 'node:assert/strict'
import http from 'node:http'
import {once} from 'node:events'
import {test} from 'node:test'
import {generate,job,workflows} from '../plugins/comfyui/api.mjs'
import {installEditor} from '../plugins/comfyui/proxy.mjs'
import {WebSocket,WebSocketServer} from 'ws'

test('ComfyUI editor transport and agent workflow share a protected backend',async () => {
  const id = 'b9a075cf-6505-463b-aeb2-6e0111b6e229'
  const checkpoint = 'v1-5-pruned-emaonly.safetensors'
  const file = '/userdata/workflows%2Fsample.json'
  const graph = {nodes:[],links:[],version:0.4}
  let queued, saved, receivedCookie, requests = 0
  const backend = http.createServer(async(req,res)=>{
    requests++
    receivedCookie = req.headers.cookie
    const body = []
    for await (const part of req) body.push(part)
    const path = req.url.split('?')[0]
    let result
    if (path === '/models/checkpoints') result=[checkpoint]
    else if (path === '/prompt') {
      queued=JSON.parse(Buffer.concat(body)).prompt
      result={prompt_id:id,number:1}
    } else if (path === '/api/jobs/'+id) {
      result={id,status:'completed',workflow:{prompt:queued},outputs:{'7':{images:[{filename:'test output.png',subfolder:'DSH',type:'output'}]}}}
    } else if (path === file && req.method === 'POST') {
      saved=JSON.parse(Buffer.concat(body))
      result='workflows/sample.json'
    } else if (path === file) result=saved
    else if (path === '/userdata') result=['sample.json']
    else {res.writeHead(404).end();return}
    res.setHeader('content-type','application/json')
    res.end(JSON.stringify(result))
  })
  const ws = new WebSocketServer({server:backend})
  ws.on('connection',peer=>peer.send('ready'))
  backend.listen(0,'127.0.0.1')
  await once(backend,'listening')
  const prev = process.env.DSH_COMFYUI_URL
  process.env.DSH_COMFYUI_URL = 'http://127.0.0.1:'+backend.address().port
  let route
  const upgrades = new Map(), disposers=[]
  const ctx={
    connection:{admit:req=>req.headers.cookie==='owner' ? {peer:{principal:{kind:'owner'}}} : {rejection:401}},
    effect:fn=>disposers.push(fn()),
    webServer:{register:r=>{route=r},registerUpgrade:r=>upgrades.set(r.path,r)}
  }
  installEditor(ctx)
  const studio = http.createServer((req,res)=>route.handler(req,res))
  studio.on('upgrade',(req,socket,head)=>upgrades.get(req.url).handler(req,socket,head))
  studio.listen(0,'127.0.0.1');await once(studio,'listening')
  const origin='http://127.0.0.1:'+studio.address().port
  try {
    assert.equal((await fetch(origin+'/comfy/models/checkpoints')).status,401)
    assert.equal(requests,0)
    const allowed=await fetch(origin+'/comfy/models/checkpoints',{headers:{cookie:'owner'}})
    assert.deepEqual(await allowed.json(),[checkpoint])
    assert.equal(receivedCookie,undefined)
    const socket=new WebSocket(origin.replace('http:','ws:')+'/comfy/ws',{headers:{cookie:'owner'}})
    assert.equal(String((await once(socket,'message'))[0]),'ready')
    socket.close();await once(socket,'close')
    const result=await generate({prompt:'Cup',seed:14})
    assert.equal(result.id,id)
    assert.equal(queued['2'].inputs.text,'Cup')
    const done=await job({id,includeGraph:true})
    assert.deepEqual(done.graph,queued)
    assert.equal(done.files[0].url,'/comfy/view?filename=test+output.png&subfolder=DSH&type=output')
    await workflows({action:'save',name:'sample.json',workflow:graph})
    assert.deepEqual(await workflows({action:'list'}),['sample.json'])
    assert.deepEqual(await workflows({action:'get',name:'sample.json'}),graph)
    await assert.rejects(workflows({action:'save',name:'../settings.json',workflow:graph}))
  } finally {
    for (const dispose of disposers) if(typeof dispose === 'function') dispose()
    studio.close();ws.close();backend.close()
    if(prev===undefined)delete process.env.DSH_COMFYUI_URL
    else process.env.DSH_COMFYUI_URL=prev
  }
})
