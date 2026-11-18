import assert from 'node:assert/strict'
import test from 'node:test'
import {mkdtemp,mkdir,symlink,rm} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {fileURLToPath} from 'node:url'
import {selectionId,parseSelection,optionArgs,saveOptions} from '../agents/options.mjs'
import {CliAdapter} from '../agents/index.mjs'
import {parseGoogleModels,googleEnvironment} from '../agents/google.mjs'

test('Google account catalog groups only actual effort variants and keeps fixed models',()=>{
  const cap = parseGoogleModels('Fetching models...\ngemini-test-high\tGemini Test (High)\ngemini-test-medium\tGemini Test (Medium)\nplanner\tOrchestrator\n')
  assert.deepEqual(cap.models,['gemini-test-medium','planner'])
  assert.deepEqual(cap.byModel['gemini-test-medium'].efforts,['medium','high'])
  assert.deepEqual(cap.byModel.planner.efforts,[])
  assert.equal(cap.variants[0].label,'Gemini Test')
  assert.throws(()=>parseGoogleModels('Authentication required'))
  assert.deepEqual(googleEnvironment({HOME:'/home/test',PATH:'/bin',DBUS_SESSION_BUS_ADDRESS:'bus',GEMINI_API_KEY:'key',GOOGLE_API_KEY:'key',AGY_ADC_AUTH:'true'}),{HOME:'/home/test',PATH:'/bin',DBUS_SESSION_BUS_ADDRESS:'bus'})
  assert.deepEqual(optionArgs('google',{model:'gemini-test-medium',effort:'high'}),['--model','gemini-test-high','--effort','high'])
})

test('explicit choices validate and remain independent of defaults and other chats',async t=>{
  const temp = await mkdtemp(join(tmpdir(),'dsh-model-options-'))
  t.after(()=>rm(temp,{recursive:true,force:true}))
  const bin = join(temp,'bin');await mkdir(bin)
  await symlink(fileURLToPath(new URL('agents.py',import.meta.url)),join(bin,'agy'))
  const old = {PATH:process.env.PATH,DSH_HOME:process.env.DSH_HOME}
  process.env.PATH=bin+':'+process.env.PATH;process.env.DSH_HOME=join(temp,'state')
  t.after(()=>{for(const [key,value] of Object.entries(old)) {if(value===undefined)delete process.env[key];else process.env[key]=value}})
  const a = selectionId('google',{model:'gemini-test-medium',effort:'high'})
  const b = selectionId('google',{model:'gemini-test-medium',effort:'low'})
  assert.deepEqual(parseSelection(a),{service:'google',options:{model:'gemini-test-medium',effort:'high'}})
  assert.throws(()=>parseSelection('unregistered'))
  assert.throws(()=>parseSelection('google@?'))
  assert.throws(()=>selectionId('google',{fast:true}))
  assert.throws(()=>selectionId('google',{model:'--bad'}))
  assert.throws(()=>selectionId('google',{effort:'ultra'}))
  const adapter = new CliAdapter()
  assert.equal((await adapter.resolveModel('local-cli',a)).id,a)
  await saveOptions('google',{model:'planner'})
  async function response(model) {
    let text=''
    for await (const block of adapter.stream({model,system:'Test',messages:[{role:'user',content:[{type:'text',text:'TEST_OPTIONS'}]}]})) {
      if(block.type==='text-delta') text+=block.text
    }
    return JSON.parse(Buffer.from(text.slice(8),'base64').toString()).args
  }
  const outputs=await Promise.all([response(a),response(b),response('google')])
  assert.equal(outputs[0][outputs[0].indexOf('--model')+1],'gemini-test-high')
  assert.equal(outputs[1][outputs[1].indexOf('--model')+1],'gemini-test-low')
  assert.equal(outputs[2][outputs[2].indexOf('--model')+1],'planner')
  assert(!outputs[2].includes('--effort'))
})
