import {selectDevin} from './devin.mjs'
import {googleEnvironment} from './google.mjs'
import {cliOptions, optionArgs} from './options.mjs'
import { execFile } from 'node:child_process'
import { constants } from 'node:fs'
import { access, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

export const agentNames = Object.freeze({
  auto: '自動引き継ぎ（Cursor → Astra → Opus）',
  google: 'Google AI Ultra',
  'devin-free': 'Devin（無料のみ）', devin: 'Devin', muse: 'Muse', grok: 'Grok', cursor: 'Cursor', codex: 'Codex', claude: 'Claude Code'
})

export async function executable(model) {
  if (!Object.hasOwn(agentNames, model)) throw new Error(`Unknown CLI model: ${model}`)
  const command = model === 'google' ? 'agy' : model === 'cursor' ? 'cursor-agent' : model === 'devin-free' ? 'devin' : model
  const dirs = [...(process.env.PATH || '').split(delimiter).filter(Boolean),
    join(homedir(), '.local/bin'), join(homedir(), '.grok/bin'), '/usr/lib/chatgpt/resources']
  const candidates = dirs.map(dir => join(dir, command))
  if (model === 'claude') {
    const base = join(homedir(), '.config/Claude/claude-code')
    const versions = await readdir(base).catch(() => [])
    for (const version of versions.sort((a, b) => b.localeCompare(a, 'en', {numeric: true}))) {
      const builds = await readdir(join(base, version)).catch(() => [])
      for (const build of builds) candidates.push(join(base, version, build, 'claude'))
    }
  }
  for (const path of candidates) {
    if (await access(path, constants.X_OK).then(() => true, () => false)) return path
  }
  throw new Error(`${agentNames[model]}: ${command} が見つかりません`)
}

function invocation(model, dir, options) {
  const args = optionArgs(model, options)
  switch (model) {
    case 'google': return [...args, '--input-format', 'stream-json', '--output-format', 'stream-json', '--disable-slash-commands', '--settings', JSON.stringify({permissions:{deny:['command(*)','read_file(*)','write_file(*)','read_url(*)','execute_url(*)','mcp(*)']}})]
    case 'devin-free':
    case 'devin': return [...args,'--process-model','in-process','--respect-workspace-trust','false','--permission-mode','normal','--verbosity','0','--prompt-file',join(dir,'request.txt'),'-p']
    case 'muse': return ['exec', ...args, '--preset', 'empty', '--no-session-log', '--json', '--disable-web-tools', '--prompt-file', join(dir, 'request.txt')]
    case 'grok': return [...args, '--no-subagents', '--disable-web-search', '--tools', '', '--prompt-file', join(dir, 'request.txt')]
    case 'cursor': return ['-p', ...args, '--trust', '--mode', 'ask', '--output-format', 'json']
    case 'codex': return ['-a', 'never', 'exec', ...args, '--ignore-user-config', '--sandbox', 'read-only', '--ephemeral', '--skip-git-repo-check', '--output-last-message', join(dir, 'answer.txt'), '--json', '-']
    case 'claude': return ['-p', ...args, '--tools', options.ultracode ? 'Workflow,Agent' : '', '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json']
  }
}

function cliEnvironment(model, options={}) {
  if (model === 'google') return googleEnvironment()
  const keep = /^(HOME|USER|LOGNAME|PATH|SHELL|LANG|LC_.*|TMPDIR|TERM|XDG_.*|DBUS_SESSION_BUS_ADDRESS|SSH_AUTH_SOCK|https?_proxy|all_proxy|no_proxy|SSL_CERT_FILE|NODE_EXTRA_CA_CERTS)$/i
  const auth = {
    devin: /^DEVIN_/, 'devin-free': /^DEVIN_/, muse: /^(MUSE_|META_)/, grok: /^(GROK_|XAI_)/, cursor: /^CURSOR_/,
    codex: /^(CODEX_HOME$|OPENAI_)/, claude: /^(ANTHROPIC_|CLAUDE_CODE_OAUTH_TOKEN$|CLAUDE_CONFIG_DIR$)/
  }[model]
  return Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !(model === 'codex' && options.subscription && /^OPENAI_/i.test(key)) && (keep.test(key) || auth.test(key))))
}

async function requireCodexSubscription() {
  // exec ignores the user configuration. Pin its file credential store and
  // check it before starting the CLI; never switch, clear, or copy credentials.
  let auth
  try {
    auth = JSON.parse(await readFile(join(process.env.CODEX_HOME || join(homedir(),'.codex'),'auth.json'),'utf8'))
  } catch {
    throw Error('サブスク用のCodex ChatGPT認証を確認できません。CodexにChatGPTでログインしてください。APIキーには切り替えません。')
  }
  if (auth.auth_mode !== 'chatgpt' || !auth.tokens?.access_token || auth.OPENAI_API_KEY)
    throw Error('サブスク用のCodex ChatGPT認証が必要です。現在の認証を変更せず停止しました。APIキーには切り替えません。')
}

function failureDetail(stdout, stderr) {
  // Structured CLIs can report authentication failures on stdout even with a nonzero status.
  const last = stdout.trimEnd().split('\n').at(-1)
  try {
    const event = JSON.parse(last)
    const message = event.error?.message ?? event.error ?? event.result?.error ?? event.message ?? event.result ?? event.payload?.reason
    if (typeof message === 'string') return message.slice(-1000)
  } catch { /* Plain-text CLI diagnostics follow. */ }
  return (stderr.trim() || last || '').split('\n').filter(Boolean).slice(-3).join('\n').slice(-1000)
}

function execute(file, args, cwd, input, env, signal) {
  signal?.throwIfAborted()
  return new Promise((resolve, reject) => {
    // Each one-shot CLI owns a process group, including launcher and worker processes.
    const proc = execFile(file, args, {cwd, env, detached: true, maxBuffer: 32 * 1024 * 1024}, (error, stdout, stderr) => {
      signal?.removeEventListener('abort', cancel)
      if (signal?.aborted) {
        reject(signal.reason)
      } else if (error) {
        if (error.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') terminateGroup('SIGKILL')
        reject(new Error(`CLI exited (${error.code ?? error.signal}): ${failureDetail(stdout, stderr)}`))
      } else resolve(stdout)
    })
    function terminateGroup(sig) {
      if (!proc.pid) return
      try { process.kill(-proc.pid, sig) } catch (error) { if (error.code !== 'ESRCH') throw error }
    }
    function cancel() {
      terminateGroup('SIGTERM')
      const killTimer = setTimeout(() => terminateGroup('SIGKILL'), 1500)
      killTimer.unref()
    }
    signal?.addEventListener('abort', cancel, {once: true})
    proc.stdin.on('error', error => { if (error.code !== 'EPIPE') reject(error) })
    proc.stdin.end(input)
  })
}

export async function runCli(model, prompt, signal, selected) {
  const file = await executable(model)
  const dir = await mkdtemp(join(tmpdir(), 'dsh-cli-'))
  try {
    const isDevin = model === 'devin' || model === 'devin-free'
    const usesFile = isDevin || model === 'muse' || model === 'grok'
    if (usesFile) await writeFile(join(dir, 'request.txt'), prompt, {mode: 0o600})
    let options = selected ?? await cliOptions(model)
    if (isDevin) options = await selectDevin(file,model,options,signal)
    if (model === 'codex' && options.subscription) await requireCodexSubscription()
    const input = model === 'google' ? JSON.stringify({event:'user',message:{content:prompt}})+'\n' : usesFile ? undefined : prompt
    const stdout = await execute(file, invocation(model, dir, options), dir, input, cliEnvironment(model,options), signal)
    let answer
    if (model === 'codex') answer = await readFile(join(dir, 'answer.txt'), 'utf8')
    else if (model === 'muse') {
      const rows = stdout.trim().split('\n').map(line => JSON.parse(line))
      const end = rows.findLast(row => row.payload_type?.startsWith('run.terminal.'))
      if (end?.payload.terminal !== 'completed') throw new Error(end?.payload.reason || 'Muse did not complete the request')
      answer = end.payload.text
    } else if (model === 'google') {
      const rows = stdout.trim().split('\n').map(line=>JSON.parse(line))
      const data = rows.findLast(row=>row.event==='result')?.result
      if (!data) throw new Error('Antigravity did not return a result')
      if (data.status !== 'SUCCESS') throw new Error(data.error || `Antigravity: ${data.status || 'invalid response'}`)
      answer = data.response
    } else if (model === 'cursor' || model === 'claude') {
      const data = JSON.parse(stdout)
      if (data.is_error || (data.subtype && data.subtype !== 'success')) throw new Error(String(data.result || data.subtype))
      answer = data.result
    } else answer = stdout
    if (typeof answer !== 'string' || !answer.trim()) throw new Error('Empty model response')
    return answer.trim()
  } catch (error) {
    signal?.throwIfAborted()
    throw new Error(`${agentNames[model]}: ${error.message}`, {cause: error})
  } finally {
    await rm(dir, {recursive: true, force: true})
  }
}
