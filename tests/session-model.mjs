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
  for (const cli of ['codex','grok','claude','muse','devin','cursor-agent']) {
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
  async function choose(name) {
    await composer.getByRole('button',{name:/モデルを選択/}).click()
    await page.getByRole('menuitemradio',{name,exact:true}).click()
    await composer.getByRole('button',{name:'モデルを選択、現在 '+name,exact:true}).waitFor()
  }
  async function firstReply(cli) {
    await composer.getByRole('textbox',{name:/作りたいもの/}).fill('TEST_CHILD')
    await composer.getByRole('button',{name:'メッセージを送信'}).click()
    await page.getByText('CHILD_OK_'+cli,{exact:true}).first().waitFor()
  }
  await choose('Grok')
  assert.equal(await page.locator('[data-conversation-session]').count(),0)
  await page.getByRole('button',{name:'ワークスペースを選択'}).click()
  await page.getByRole('button',{name:'ワークスペースを選択'}).filter({hasText:'Studio folder'}).waitFor()
  await firstReply('grok')
  await page.getByRole('button',{name:'新規セッション',exact:true}).first().click()
  await choose('Claude Code')
  await firstReply('claude')
  await page.reload()
  await composer.getByRole('button',{name:'モデルを選択、現在 Claude Code'}).waitFor()
  assert.deepEqual(errors,[])
  console.log('Composer menu routes the first message before and after workspace selection and survives reload.')
} finally {
  server?.kill()
  await browser?.close()
  await stopped
  await rm(temp,{recursive:true,force:true})
}
