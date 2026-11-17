// Exercises the real Harness gateway with a temporary Access signing key and isolated local state.
import assert from 'node:assert/strict'
import {spawn} from 'node:child_process'
import {once} from 'node:events'
import {mkdtemp,rm} from 'node:fs/promises'
import http from 'node:http'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {createInterface} from 'node:readline'
import {fileURLToPath} from 'node:url'
import WebSocket from 'ws'
import {generateKeyPair,SignJWT,createLocalJWKSet,exportJWK} from 'jose'
import {accessVerifier,gateway,harnessCookie} from '../deploy/gateway.mjs'

const state=await mkdtemp(join(tmpdir(),'darask-gateway-'))
const {publicKey,privateKey}=await generateKeyPair('RS256')
const keys=createLocalJWKSet({keys:[{...await exportJWK(publicKey),alg:'RS256',kid:'fixture'}]})
const host='studio.example.test',issuer='https://fixture.cloudflareaccess.com',aud='fixture'
const launch=fileURLToPath(new URL('../dsh',import.meta.url))
const child=spawn(launch,['web','--port','33662','--no-open','--host','127.0.0.1','--public-url','https://'+host,'--trusted-host',host],
  {env:{...process.env,OPENAI_API_KEY:'catalog-fixture',DSH_HOME:state},stdio:['ignore','pipe','pipe']})
const stopped=once(child,'exit')
let server
const timer=setTimeout(()=>child.kill(),45000)
try{
  const token=await new Promise((resolve,reject)=>{
    for(const stream of [child.stdout,child.stderr])createInterface({input:stream}).on('line',line=>{
      const match=line.match(/[?&]token=([\w-]+)/)
      if(match)resolve(match[1])
    })
    child.once('exit',()=>reject(Error('Harness failed to start')))
  })
  const cookie=await harnessCookie(33662,host,token)
  server=gateway({host,port:33662,verify:accessVerifier(issuer,aud,keys),cookie:async()=>cookie})
  server.listen(0,'127.0.0.1')
  await once(server,'listening')
  function request(path,jwt,body,extra={}){
    return new Promise((resolve,reject)=>{
      const headers={host,...jwt&&{'cf-access-jwt-assertion':jwt},...body&&{'content-type':'application/json'},...extra}
      const req=http.request({hostname:'127.0.0.1',port:server.address().port,path,method:body?'POST':'GET',headers},res=>{
        let data='';res.on('data',chunk=>data+=chunk);res.on('end',()=>resolve([res.statusCode,data]))
      })
      req.on('error',reject);req.end(body&&JSON.stringify(body))
    })
  }
  const sign=(audience,exp)=>new SignJWT({sub:'fixture'}).setProtectedHeader({alg:'RS256',kid:'fixture'}).setIssuer(issuer).setAudience(audience).setIssuedAt().setExpirationTime(exp).sign(privateKey)
  const valid=await sign(aud,'2m')
  for(const invalid of [undefined,await sign('other','2m'),await sign(aud,Math.floor(Date.now()/1000)-5)]){
    assert.equal((await request('/',invalid))[0],403)
  }
  assert.equal((await request('/',valid,null,{origin:'https://other.invalid'}))[0],403)
  const [status,html]=await request('/',valid)
  assert.equal(status,200);assert(html.includes('<html'))
  const method='session/modelCatalog'
  const [code,raw]=await request('/api/'+method,valid,{type:'client-request',rpcId:'check',method,payload:{args:{}}},{origin:'https://'+host})
  assert.equal(code,200)
  const groups=JSON.parse(raw).result.value.groups
  assert(groups.some(g=>g.id==='openai'&&g.models.length))
  assert(groups.some(g=>g.id==='local-cli'&&g.models.some(m=>m.id==='codex')))
  const socket=new WebSocket('ws://127.0.0.1:'+server.address().port+'/api/remote.mux',{headers:{host,'cf-access-jwt-assertion':valid,origin:'https://'+host}})
  await once(socket,'open');socket.close();await once(socket,'close')
  console.log('Verified WebSocket, JWT acceptance and rejection, protected page and RPC. OpenAI API models:',groups.find(g=>g.id==='openai').models.length)
}finally{
  clearTimeout(timer);server?.close();child.kill();await stopped;await rm(state,{recursive:true,force:true})
}
