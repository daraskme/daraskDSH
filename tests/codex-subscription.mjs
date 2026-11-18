import test from 'node:test'
import assert from 'node:assert/strict'
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {execFile} from 'node:child_process'
import {promisify} from 'node:util'
import {selectionId,parseSelection,validate} from '../agents/options.mjs'

const run=promisify(execFile)
const cliUrl=new URL('../agents/cli.mjs',import.meta.url).href
const subscription={model:'gpt-6.2-astra',effort:'high',subscription:true}

async function fixture(t,auth) {
  const directory=await mkdtemp(join(tmpdir(),'dsh-subscription-'))
  t.after(()=>rm(directory,{recursive:true,force:true}))
  const bin=join(directory,'bin'),fixtureCodexHome=join(directory,'auth')
  await mkdir(bin);await mkdir(fixtureCodexHome)
  const authPath=join(fixtureCodexHome,'auth.json')
  await writeFile(authPath,JSON.stringify(auth))
  await writeFile(join(bin,'codex'),`#!/usr/bin/python3
import json,os,pathlib,sys
out={'args':sys.argv[1:],'apiKeyPresent':'OPENAI_API_KEY' in os.environ,
     'apiBasePresent':'OPENAI_BASE_URL' in os.environ,
     'codexApiKeyPresent':'CODEX_API_KEY' in os.environ}
pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(out))
`,{mode:0o700})
  const env={...process.env,PATH:bin+':'+process.env.PATH,CODEX_HOME:fixtureCodexHome,
    OPENAI_API_KEY:'fixture-only',OPENAI_BASE_URL:'https://fixture.invalid',CODEX_API_KEY:'fixture-only'}
  const invoke=options=>run(process.execPath,['--input-type=module','-e',
    `import {runCli} from ${JSON.stringify(cliUrl)};
     try {console.log(await runCli('codex','fixture',undefined,JSON.parse(process.argv[1])))}
     catch(error) {console.log(JSON.stringify({error:error.message}));process.exitCode=1}`,
    JSON.stringify(options)],{env})
  return {invoke,authPath}
}

test('subscription is a validated Codex-only selection',()=>{
  assert.deepEqual(parseSelection(selectionId('codex',subscription)),{service:'codex',options:subscription})
  assert.throws(()=>validate('cursor',{subscription:true}),/利用できない/)
  assert.throws(()=>validate('codex',{subscription:'true'}),/利用できない/)
})

test('subscription pins ChatGPT auth and suppresses API credentials and endpoint overrides',async t=>{
  const {invoke,authPath}=await fixture(t,{auth_mode:'chatgpt',tokens:{access_token:'fixture-only'},OPENAI_API_KEY:null})
  const original=await readFile(authPath,'utf8')
  const {stdout}=await invoke(subscription)
  const value=JSON.parse(stdout)
  for(const flag of ['forced_login_method="chatgpt"','model_provider="openai"',
    'cli_auth_credentials_store="file"','model_reasoning_effort="high"']) assert(value.args.includes(flag))
  assert.equal(value.args[value.args.indexOf('--model')+1],'gpt-6.2-astra')
  assert.equal(value.apiKeyPresent,false);assert.equal(value.apiBasePresent,false)
  assert.equal(value.codexApiKeyPresent,false)
  assert.equal(await readFile(authPath,'utf8'),original)
})

test('API-only auth fails before CLI execution without modifying its stored credentials',async t=>{
  const {invoke,authPath}=await fixture(t,{auth_mode:'apikey',OPENAI_API_KEY:'fixture-only'})
  const original=await readFile(authPath,'utf8')
  await assert.rejects(invoke(subscription),error=>{
    const {error:message}=JSON.parse(error.stdout)
    assert.match(message,/ChatGPT認証が必要/)
    assert(!message.includes('fixture-only'))
    return true
  })
  assert.equal(await readFile(authPath,'utf8'),original)
})

test('ordinary Codex selections preserve existing API authentication behavior',async t=>{
  const {invoke}=await fixture(t,{auth_mode:'apikey',OPENAI_API_KEY:'fixture-only'})
  const value=JSON.parse((await invoke({model:'gpt-6.2-astra'})).stdout)
  assert.equal(value.apiKeyPresent,true)
  assert.equal(value.apiBasePresent,true)
  assert(!value.args.some(arg=>arg.startsWith('forced_login_method=')))
})
