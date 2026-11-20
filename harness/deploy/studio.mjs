import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { accessVerifier, gateway, harnessCookie } from './gateway.mjs'
const host = process.env.STUDIO_HOST
const port = Number(process.env.STUDIO_HARNESS_PORT || 3092)
const listener = Number(process.env.STUDIO_PORT || 8789)
const issuer = process.env.CF_ACCESS_ISSUER
if (!/^[a-z0-9.-]+$/.test(host || '') || !/^https:\/\/[a-z0-9-]+\.cloudflareaccess\.com$/.test(issuer || '')) throw Error('Invalid studio configuration')
const verify = accessVerifier(issuer,process.env.CF_ACCESS_AUD)
const child = spawn(fileURLToPath(new URL('../dsh',import.meta.url)),['web','--no-open','--host','127.0.0.1','--port',String(port),'--trusted-host',host,'--public-url','https://'+host],{stdio:['ignore','pipe','pipe']})
let server,starting=false,stopping=false
const deadline=setTimeout(()=>{console.error('Harness startup timed out');child.kill()},60000)
for (const stream of [child.stdout,child.stderr]) createInterface({input:stream}).on('line',async line=>{
  const match=line.match(/[?&]token=([\w-]{32,})/)
  if(!match || starting)return
  starting=true
  try {
    const exchange=()=>harnessCookie(port,host,match[1])
    let signed=await exchange(),renew=Date.now()+3600000,pending
    const cookie=async()=>{
      if(Date.now()<renew)return signed
      pending ||= exchange().then(v=>{signed=v;renew=Date.now()+3600000}).finally(()=>{pending=undefined})
      await pending
      return signed
    }
    server=gateway({host,port,verify,cookie})
    server.on('error',err=>{console.error('Studio listener:',err.code);child.kill()})
    server.listen(listener,'127.0.0.1',()=>{clearTimeout(deadline);console.log('Studio ready on loopback port '+listener)})
  } catch(e) { console.error(e.message);child.kill() }
})
child.on('error',()=>process.exit(1))
child.on('exit',()=>{clearTimeout(deadline);server?.close();process.exit(stopping ? 0 : 1)})
for(const sig of ['SIGTERM','SIGINT'])process.on(sig,()=>{stopping=true;server?.close();child.kill()})
