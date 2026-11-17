import http from 'node:http'
import {createRemoteJWKSet,jwtVerify} from 'jose'

export function accessVerifier(issuer, audience, keys=createRemoteJWKSet(new URL('/cdn-cgi/access/certs',issuer))) {
  if (!audience || !issuer) throw Error('Cloudflare Access configuration missing')
  return async token => {
    const {payload} = await jwtVerify(token,keys,{issuer,audience,algorithms:['RS256'],requiredClaims:['exp','iat']})
    return payload.exp
  }
}
export async function harnessCookie(port,host,token) {
  return new Promise((resolve,reject)=>{
    const req = http.get({hostname:'127.0.0.1',port,path:'/?token='+encodeURIComponent(token),headers:{host}},res=>{
      res.resume()
      const cookie = res.headers['set-cookie']?.find(v=>v.startsWith('dsh-auth-'))
      if(!cookie || res.statusCode<300 || res.statusCode>399) reject(Error('Harness login exchange failed: '+res.statusCode))
      else resolve(cookie.split(';',1)[0])
    })
    req.setTimeout(6000,()=>req.destroy(Error('Harness login exchange timed out')))
    req.on('error',reject)
  })
}
export function gateway({host,port,verify,cookie}) {
  const origin = 'https://'+host
  async function target(req,upgrade) {
    if (req.headers.host?.toLowerCase() !== host || !req.url?.startsWith('/') || req.url.startsWith('//')) throw Error('host')
    if (req.headers.origin && req.headers.origin !== origin) throw Error('origin')
    if (req.headers['sec-fetch-site']==='cross-site' && (upgrade || !['HEAD','GET'].includes(req.method))) throw Error('origin')
    const expires = await verify(req.headers['cf-access-jwt-assertion'])
    const headers = {host, cookie:await cookie()}
    for (const [name,value] of Object.entries(req.headers)) {
      if (/^(accept($|-)|content-(?:type|length|range|encoding)$|range$|if-(?:range|modified-since|none-match)$|user-agent$|origin$|sec-(?:fetch-|websocket-)|last-event-id$|x-dsh-)/i.test(name)) headers[name] = value
    }
    if (upgrade) {headers.upgrade='websocket';headers.connection='Upgrade'}
    return {expires,options:{hostname:'127.0.0.1',port,path:req.url,method:req.method,headers}}
  }
  const server = http.createServer(async(req,res)=>{
    let upstream
    try { upstream = await target(req,false) }
    catch { res.writeHead(403,{'cache-control':'no-store'});res.end('Cloudflare Access authentication required');return }
    const client = http.request(upstream.options,response=>{
      const headers = {...response.headers}
      delete headers['set-cookie']
      res.writeHead(response.statusCode,headers)
      response.on('error',()=>res.destroy())
      response.pipe(res)
    })
    client.on('error',()=>{if(!res.headersSent)res.writeHead(502);res.end()})
    res.on('close',()=>client.destroy())
    req.pipe(client)
  })
  server.on('upgrade',async(req,socket,head)=>{
    socket.on('error',()=>socket.destroy())
    let upstream
    try { upstream = await target(req,true) }
    catch { socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');return }
    const client = http.request(upstream.options)
    client.on('upgrade',(response,backend,extra)=>{
      const headers = ['HTTP/1.1 101 Switching Protocols']
      for (const [key,value] of Object.entries(response.headers)) {
        if (key==='connection' || key==='upgrade' || key.startsWith('sec-websocket-')) headers.push(key+': '+value)
      }
      socket.write(headers.join('\r\n')+'\r\n\r\n')
      if(head.length)backend.write(head)
      if(extra.length)socket.write(extra)
      socket.pipe(backend).pipe(socket)
      const timer = setTimeout(()=>socket.destroy(),Math.min(2147483647,Math.max(1,upstream.expires*1000-Date.now())))
      socket.on('close',()=>{clearTimeout(timer);backend.destroy()})
      backend.on('close',()=>socket.destroy())
      backend.on('error',()=>socket.destroy())
    })
    client.on('response',response=>{response.resume();socket.end('HTTP/1.1 '+response.statusCode+' Upstream response\r\nConnection: close\r\n\r\n')})
    client.on('error',()=>socket.destroy())
    client.end()
  })
  return server
}
