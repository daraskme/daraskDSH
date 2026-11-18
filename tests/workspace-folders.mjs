// Exercise the real Studio composer, DSH model adapter and confined write tool.
import assert from 'node:assert/strict'
import {execFileSync, spawn} from 'node:child_process'
import {once} from 'node:events'
import {mkdtemp,mkdir,readFile,rm,symlink,writeFile} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {createInterface} from 'node:readline'
import {fileURLToPath} from 'node:url'
import {Session, appendPluginRecord} from '@deepseek-ai/dsh-session'
import {canonicalFolders,policyWithFolders,projection} from '../plugins/workspace-folders/index.mjs'

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const dsh = fileURLToPath(new URL('../dsh',import.meta.url))
const state = await mkdtemp(join(tmpdir(),'dsh-folders-'))
// System temp directories are writable even without a folder grant. Keep the
// write targets elsewhere so a successful write actually tests the new policy.
const files = await mkdtemp(fileURLToPath(new URL('../.workspace-test-',import.meta.url)))
const primary = join(files,'primary'), second = join(files,'資料 B'), third = join(files,'C')
const output = join(second,'proof.txt')
const bin = join(state,'bin')
const env = {...process.env, DSH_HOME:join(state,'state'), DARASK_DSH_MODEL:'cursor', PATH:bin+':'+process.env.PATH}
let server, exited, browser
try {
  await Promise.all([primary,second,third,bin].map(path=>mkdir(path)))
  await symlink(second,join(files,'alias'))
  assert.deepEqual(await canonicalFolders([primary,second,join(files,'alias'),third],primary),[second,third])
  assert.equal(policyWithFolders({mode:'read-only'},[second]).additionalWritableRoots,undefined)
  const owner = Session.create('test-owner')
  appendPluginRecord(owner,'plugin:darask/workspace-folders',{owner:owner.id,folders:[second,third]})
  const history = owner.snapshotEvents()
  assert.deepEqual(history.reduce(projection.apply,projection.init(owner.header)).folders,[second,third])
  const fork = Session.create('another-chat')
  assert.deepEqual(history.reduce(projection.apply,projection.init(fork.header)).folders,[])
  await writeFile(join(bin,'xdg-user-dir'),'#!/bin/sh\nexit 1\n',{mode:0o755})
  await writeFile(join(bin,'cursor-agent'),`#!/usr/bin/env python3
import json,sys
if '--list-models' in sys.argv:
    print('auto - Auto')
    sys.exit()
req=json.loads(sys.stdin.read().splitlines()[-1])
msgs=req['messages']
turn=max(i for i,m in enumerate(msgs) if m['role']=='user' and 'FOLDER_PROBE' in str(m['content']))
seen=any('Additional workspace folders selected for this chat' in str(m['content']) and ${JSON.stringify(third)} in str(m['content']) for m in msgs)
results=[m for m in msgs[turn+1:] if m['role']=='tool']
if results:
    answer='WRITE_DENIED' if results[-1].get('isError') or 'denied' in str(results[-1]['content']).lower() else ('WRITE_OK_WITH_CONTEXT' if seen else 'WRITE_OK_NO_CONTEXT')
else:
    answer=json.dumps({'dsh_tool':'write','arguments':{'file_path':${JSON.stringify(output)},'content':'hello from the same chat'}})
print(json.dumps({'type':'result','subtype':'success','result':answer}))
`,{mode:0o755})
  execFileSync(dsh,['workspace','setup'],{env})
  execFileSync(dsh,['workspace','setup'],{env})
  server = spawn(dsh,['web','--no-open','--host','127.0.0.1','--port','0','--trusted-host','studio.example.test'],{env,cwd:primary,stdio:['ignore','pipe','pipe']})
  exited = once(server,'exit')
  const url = await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('DSH startup timed out')),25000)
    for (const source of [server.stdout,server.stderr]) createInterface({input:source}).on('line',line=>{
      const match=line.match(/http:\/\/127\.0\.0\.1:\d+\/\?token=[\w-]+/)
      if(match){clearTimeout(timer);resolve(match[0])}
      else if (/error|warn/i.test(line)) console.error(line.slice(0,600))
    })
    exited.then(()=>{clearTimeout(timer);reject(Error('DSH exited'))})
  })
  browser = await chromium.launch({executablePath:process.env.CHROME_BIN})
  const page = await browser.newPage({locale:'ja-JP',userAgent:'Mozilla/5.0 (X11; Linux x86_64) Chrome/130.0.0.0 Safari/537.36'})
  await page.route('**/api/darask.usage',route=>route.fulfill({json:{}}))
  const picks = [primary,second,null]
  await page.route('**/api/darask.pickWorkspace',route=>route.fulfill({json:{path:picks.shift()}}))
  await page.goto(url)
  await page.getByRole('button',{name:'ワークスペースを選択'}).click()
  page.on('pageerror',e=>console.error('pageerror',e.message))
  const toggle = page.getByRole('button',{name:'作業フォルダ',exact:true})
  await toggle.click({timeout:10000}).catch(async e=>{console.error(await page.locator('body').innerText());throw e})
  const dialog = page.getByRole('dialog',{name:'このチャットの作業フォルダ'})
  const field = dialog.getByLabel('フォルダの絶対パス')
  await dialog.getByRole('button',{name:'フォルダを選択…'}).click()
  await dialog.getByText(second,{exact:true}).waitFor()
  await field.fill(third)
  await dialog.getByRole('button',{name:'追加',exact:true}).click()
  await dialog.getByRole('button',{name:'保存',exact:true}).click()
  await dialog.waitFor({state:'hidden'})
  await page.reload()
  await toggle.click()
  await dialog.getByText(second,{exact:true}).waitFor()
  await dialog.getByText(third,{exact:true}).waitFor()
  await dialog.getByRole('button',{name:'キャンセル'}).click()
  const composer=page.locator('[data-composer-card]')
  // The Studio draft model defaults to Codex independently of the host env.
  await composer.getByRole('button',{name:/モデルを選択/}).click()
  const models=page.getByRole('dialog',{name:'LLMサービス・モデル選択'})
  await models.getByLabel('使用するLLMサービス',{exact:true}).selectOption('cursor')
  await models.getByRole('button',{name:'適用',exact:true}).click()
  async function submit() {
    await composer.locator('[contenteditable=true]').fill('FOLDER_PROBE')
    await composer.getByRole('button',{name:'メッセージを送信'}).click()
  }
  await submit()
  await page.getByText('WRITE_OK_WITH_CONTEXT',{exact:true}).first().waitFor()
  assert.equal(await readFile(output,'utf8'),'hello from the same chat')
  await toggle.click()
  await dialog.getByRole('button',{name:'フォルダを選択…'}).click()
  await dialog.locator('.dsh-folders-row').filter({hasText:second}).getByRole('button',{name:'外す'}).click()
  assert.equal(picks.length,0)
  await dialog.getByRole('button',{name:'保存',exact:true}).click()
  await dialog.waitFor({state:'hidden'})
  await rm(output)
  await submit()
  await page.getByText('WRITE_DENIED',{exact:true}).first().waitFor()
  await assert.rejects(readFile(output),{code:'ENOENT'})
  console.log('Studio multi-folder selection, reload, model context, real write and revocation passed.')
} finally {
  await browser?.close()
  server?.kill()
  if(exited) await exited
  await Promise.all([state,files].map(dir=>rm(dir,{recursive:true,force:true})))
}
