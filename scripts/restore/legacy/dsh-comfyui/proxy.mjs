import http from 'node:http'

export function comfyOrigin() {
  const url = new URL(process.env.DSH_COMFYUI_URL || 'http://127.0.0.1:8188')
  if (url.protocol !== 'http:' || !['127.0.0.1','[::1]','localhost'].includes(url.hostname) ||
      url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
    throw Error('DSH_COMFYUI_URL must be a loopback HTTP origin')
  }
  return url
}
function upstream(req, upgrade) {
  const origin = comfyOrigin()
  const path = req.url.slice('/comfy'.length)
  const headers = {}
  for (const [name,value] of Object.entries(req.headers)) {
    if (/^(accept(?:-.+)?|content-type|content-length|range|if-none-match|if-modified-since|comfy-user|sec-websocket-(key|version|protocol|extensions))$/.test(name)) headers[name] = value
  }
  if (upgrade) Object.assign(headers,{connection:'Upgrade',upgrade:'websocket'})
  // The browser's original Origin was already checked by DSH. ComfyUI applies its own loopback origin check.
  if (req.headers.origin) headers.origin = origin.origin
  return http.request(origin,{method:req.method,path,headers})
}
function rejection(connection, req) {
  const result = connection.admit(req)
  return result.rejection || (result.peer.principal.kind === 'owner' ? 0 : 403)
}
export function installEditor(ctx) {
  const {connection,webServer} = ctx
  const sockets = new Set()
  ctx.effect(()=>webServer.register({kind:'prefix',path:'/comfy',handler(req,res) {
    const status = rejection(connection,req)
    if (status) { res.writeHead(status).end(); return }
    if (req.url.split('?')[0] === '/comfy') { res.writeHead(308,{location:'/comfy/'}).end(); return }
    const client = upstream(req,false)
    client.on('response',response=>{
      const headers = {...response.headers}
      delete headers['set-cookie']
      if (headers.location?.startsWith('/')) headers.location = '/comfy'+headers.location
      res.writeHead(response.statusCode,headers)
      response.on('error',()=>res.destroy())
      response.pipe(res)
    })
    client.on('error',()=>{
      if (res.headersSent) res.destroy()
      else res.writeHead(502,{'content-type':'text/plain; charset=utf-8'}).end('ComfyUI に接続できません。')
    })
    res.on('close',()=>client.destroy())
    req.pipe(client)
  }}),'ComfyUI editor')
  for (const path of ['/comfy/ws','/comfy/api/ws']) {
    ctx.effect(()=>webServer.registerUpgrade({path,handler(req,socket,head) {
      const status = rejection(connection,req)
      if (status) { socket.end('HTTP/1.1 '+status+' Rejected\r\nConnection: close\r\n\r\n'); return }
      const client = upstream(req,true)
      socket.on('error',()=>socket.destroy())
      socket.once('close',()=>client.destroy())
      client.on('error',()=>socket.destroy())
      client.on('response',response=>{response.resume();socket.end('HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n')})
      client.on('upgrade',(response,backend,extra)=>{
        sockets.add(socket)
        const fields = ['HTTP/1.1 101 Switching Protocols']
        for (const [name,value] of Object.entries(response.headers)) {
          if (/^(connection|upgrade|sec-websocket-.+)$/.test(name)) fields.push(name+': '+value)
        }
        socket.write(fields.join('\r\n')+'\r\n\r\n')
        if (head.length) backend.write(head)
        if (extra.length) socket.write(extra)
        socket.once('close',()=>{sockets.delete(socket);backend.destroy()})
        backend.once('close',()=>socket.destroy())
        backend.on('error',()=>socket.destroy())
        socket.pipe(backend).pipe(socket)
      })
      client.end()
    }}),'ComfyUI websocket '+path)
  }
  ctx.effect(()=>()=>{for (const socket of sockets) socket.destroy()},'ComfyUI websocket cleanup')
}
