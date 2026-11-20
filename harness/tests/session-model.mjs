// Run with Node 24 and Playwright. PLAYWRIGHT_MODULE and CHROME_BIN may point to a local installation.
import assert from 'node:assert/strict'
import {spawn} from 'node:child_process'
import {once} from 'node:events'
import {mkdir, mkdtemp, rm, symlink, writeFile} from 'node:fs/promises'
import {createServer} from 'node:net'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {createInterface} from 'node:readline'
import {fileURLToPath} from 'node:url'

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const temp = await mkdtemp(join(tmpdir(),'studio-model-'))
const folder = join(temp,'Studio folder')
const bin = join(temp,'bin')
let server, browser, stopped
try {
  await mkdir(folder)
  await mkdir(bin)
  // A host without a configured Documents folder stays on the workspace-less composer.
  await writeFile(join(bin,'xdg-user-dir'),'#!/bin/sh\nexit 1\n',{mode:0o755})
  for (const cli of ['codex','grok','claude','muse','devin','cursor-agent','agy']) {
    await symlink(fileURLToPath(new URL('agents.py',import.meta.url)),join(bin,cli))
  }
  const probe = createServer().listen(0,'127.0.0.1')
  await once(probe,'listening')
  const port = probe.address().port
  await new Promise(resolve => probe.close(resolve))
  server = spawn(fileURLToPath(new URL('../dsh',import.meta.url)),[
    'web','--no-open','--port',String(port),'--trusted-host','studio.example.test'
  ],{
    cwd:temp, env:{...process.env,DSH_HOME:join(temp,'state'),PATH:bin+':'+process.env.PATH,OPENAI_API_KEY:'catalog-fixture'},
    stdio:['ignore','pipe','pipe']
  })
  stopped = once(server,'exit')
  const endpoint = await new Promise((resolve,reject) => {
    const timer = setTimeout(() => reject(Error('Harness startup timed out')),20000)
    for (const stream of [server.stdout,server.stderr]) createInterface({input:stream}).on('line',line => {
      const match = line.match(/http:\/\/127\.0\.0\.1:\d+\/\?token=[\w-]+/)
      if (match) { clearTimeout(timer); resolve(match[0]) }
    })
    stopped.then(() => { clearTimeout(timer); reject(Error('Harness exited')) })
  })
  browser = await chromium.launch({executablePath:process.env.CHROME_BIN})
  const page = await browser.newPage({locale:'ja-JP',userAgent:'Mozilla/5.0 (X11; Linux x86_64) Chrome/130.0.0.0 Safari/537.36'})
  const errors = []
  page.on('pageerror',error => errors.push(error.message))
  await page.route('**/api/darask.usage',route => route.fulfill({json:{}}))
  await page.route('**/api/darask.pickWorkspace',route => route.fulfill({json:{path:folder}}))
  await page.goto(endpoint)
  const composer = page.locator('[data-composer-card]')
  async function choose(service,model='',effort='') {
    await composer.getByRole('button',{name:/モデルを選択/}).click()
    const dialog = page.getByRole('dialog',{name:'LLMサービス・モデル選択'})
    await dialog.getByLabel('使用するLLMサービス',{exact:true}).selectOption(service)
    if (model) await dialog.getByLabel('モデルの種類',{exact:true}).selectOption(model)
    if (effort) await dialog.getByLabel('エフォート',{exact:true}).selectOption(effort)
    await dialog.getByRole('button',{name:'適用',exact:true}).click()
    await dialog.waitFor({state:'hidden'})
  }
  async function firstReply(cli) {
    await composer.locator('[contenteditable=true]').fill('TEST_CHILD').catch(async e=>{console.error(await page.locator('body').innerText());throw e})
    await composer.getByRole('button',{name:'メッセージを送信'}).click()
    await page.getByText('CHILD_OK_'+cli,{exact:true}).first().waitFor()
  }
  await choose('grok','grok-4.7','high')
  assert.equal(await page.locator('[data-conversation-session]').count(),0)
  await page.getByRole('button',{name:'ワークスペースを選択'}).click()
  await page.getByRole('button',{name:'ワークスペースを選択'}).filter({hasText:'Studio folder'}).waitFor()
  await firstReply('grok')
  const grokSession = await page.locator('[data-conversation-session]').getAttribute('data-conversation-session')
  await page.getByRole('button',{name:'新規セッション',exact:true}).first().click()
  await choose('claude','sonnet','low')
  await firstReply('claude')
  await page.reload()
  await composer.getByRole('button',{name:'モデルを選択、現在 Claude Code · sonnet · low'}).waitFor()
  await choose('google','gemini-test-medium','high')
  await composer.locator('[contenteditable=true]').fill('TEST_OPTIONS')
  await composer.getByRole('button',{name:'メッセージを送信'}).click()
  const output = page.getByText(/^OPTIONS /).last()
  await output.waitFor().catch(async e=>{console.error(await page.locator('body').innerText());throw e})
  const result = JSON.parse(Buffer.from((await output.textContent()).slice(8),'base64').toString())
  assert.equal(result.service,'google')
  assert.equal(result.args[result.args.indexOf('--model')+1],'gemini-test-high')
  assert.equal(result.args[result.args.indexOf('--effort')+1],'high')
  await page.reload()
  await composer.getByRole('button',{name:'モデルを選択、現在 Google AI Ultra · gemini-test-medium · high'}).waitFor()
  await page.getByRole('button',{name:'設定',exact:true}).click()
  await page.getByRole('button',{name:'LLMサービス',exact:true}).click()
  await page.getByLabel('使用するLLMサービス',{exact:true}).selectOption('google')
  await page.getByLabel('モデルの種類',{exact:true}).selectOption('gemini-test-medium')
  await page.getByLabel('エフォート',{exact:true}).selectOption('low')
  await page.getByRole('button',{name:'保存',exact:true}).click()
  await page.getByText('既定値を保存しました',{exact:true}).waitFor()
  await page.getByRole('button',{name:'閉じる',exact:true}).click()
  await composer.getByRole('button',{name:'モデルを選択、現在 Google AI Ultra · gemini-test-medium · high'}).waitFor()
  const googleSession = await page.locator('[data-conversation-session]').getAttribute('data-conversation-session')
  await page.locator('[data-row-key="session:'+grokSession+'"]').click()
  await composer.getByRole('button',{name:'モデルを選択、現在 Grok · grok-4.7 · high'}).waitFor()
  await page.locator('[data-row-key="session:'+googleSession+'"]').click()
  await composer.getByRole('button',{name:'モデルを選択、現在 Google AI Ultra · gemini-test-medium · high'}).waitFor()
  await page.setViewportSize({width:390,height:844})
  await composer.getByRole('button',{name:/モデルを選択/}).click()
  const mobileDialog = page.getByRole('dialog',{name:'LLMサービス・モデル選択'})
  await mobileDialog.getByLabel('使用するLLMサービス',{exact:true}).selectOption('google')
  assert(await mobileDialog.evaluate(e=>e.getBoundingClientRect().width <= innerWidth))
  if(process.env.DSH_UI_SCREENSHOT) await page.screenshot({path:process.env.DSH_UI_SCREENSHOT})
  await mobileDialog.getByLabel('使用するLLMサービス',{exact:true}).selectOption('openai')
  assert((await mobileDialog.getByLabel('モデルの種類',{exact:true}).locator('option').count())>2)
  await mobileDialog.getByRole('button',{name:'キャンセル'}).click()
  assert.deepEqual(errors,[])
  console.log('Service/model/effort selection, mid-chat switching, independent saved chats, settings defaults, OpenAI catalog and mobile layout passed.')
} finally {
  server?.kill()
  await browser?.close()
  await stopped
  await rm(temp,{recursive:true,force:true})
}
