// Owner-only API behind the "Anima LoRA" sidebar panel: scan a folder, recommend
// settings, then caption with Gemma and train natively in one background job.
import {spawn} from 'node:child_process'
import {existsSync} from 'node:fs'
import {mkdir, readdir, readFile, realpath, stat, writeFile} from 'node:fs/promises'
import {homedir} from 'node:os'
import {basename, dirname, extname, isAbsolute, join} from 'node:path'
import {fileURLToPath} from 'node:url'
import {CAPTION_INSTRUCTIONS, PURPOSES, recommend} from './presets.mjs'

const ROOT = fileURLToPath(new URL('../..', import.meta.url))
const PIPELINE = fileURLToPath(new URL('./pipeline.py', import.meta.url))
const PYTHON = process.env.DSH_ANIMA_PYTHON || join(ROOT, '.runtime/lora-venv/bin/python')
const JOBS = join(ROOT, '.runtime/anima-native/jobs')
const OUTPUT_DIR = join(ROOT, 'plugins/dsh-comfyui/loras/trained')
const IMAGES = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'])

let current = null  // {id, dir, process, exit, cancelled}

async function scan(folder) {
  if (typeof folder !== 'string' || !isAbsolute(folder.trim())) throw Error('フォルダを絶対パスで指定してください')
  const root = await realpath(folder.trim())
  if (!(await stat(root)).isDirectory()) throw Error('フォルダを指定してください')
  const images = [], captioned = []
  for (const entry of (await readdir(root, {withFileTypes:true})).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isFile() || !IMAGES.has(extname(entry.name).toLowerCase())) continue
    const path = join(root, entry.name)
    images.push(path)
    const caption = path.slice(0, -extname(path).length) + '.txt'
    if (existsSync(caption) && (await readFile(caption, 'utf8')).trim()) captioned.push(path)
  }
  if (!images.length) throw Error('フォルダに画像（png・jpg・webp など）がありません')
  return {root, images, captioned:captioned.length}
}

function checkTrigger(trigger) {
  const value = typeof trigger === 'string' ? trigger.trim() : ''
  if (value.length > 100 || /[\u0000-\u001f]/.test(value)) throw Error('トリガーワードは改行なし・100 文字以内にしてください')
  return value
}

function outputName(name, trigger, root, purpose) {
  const base = (typeof name === 'string' && name.trim()) || trigger || basename(root) + '-' + purpose
  const stem = base.replace(/\.safetensors$/i, '').replace(/[^\w.@-]+/g, '_').replace(/^[._]+|[._]+$/g, '').slice(0, 80)
  if (!stem) throw Error('出力名を指定してください')
  return stem
}

async function plan(body) {
  const {root, images, captioned} = await scan(body.folder)
  const trigger = checkTrigger(body.trigger)
  const settings = recommend(body.purpose, images.length)
  const name = outputName(body.name, trigger, root, body.purpose)
  const output = join(OUTPUT_DIR, name + '.safetensors')
  return {folder:root, images:images.length, captioned, trigger, name, output, comfyName:'trained/' + name + '.safetensors',
          exists:existsSync(output), settings, overwrite:body.overwrite !== false}
}

/** Subfolders of `path` (default: home) with their image counts, for the folder browser. */
async function list(path) {
  const root = await realpath(typeof path === 'string' && isAbsolute(path) ? path : homedir())
  const dirs = []
  for (const entry of (await readdir(root, {withFileTypes:true})).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue
    const child = join(root, entry.name)
    let images = 0
    try { images = (await readdir(child)).filter(name => IMAGES.has(extname(name).toLowerCase())).length } catch { continue }
    dirs.push({name:entry.name, path:child, images})
  }
  const files = (await readdir(root)).filter(name => IMAGES.has(extname(name).toLowerCase())).length
  return {path:root, parent:dirname(root) === root ? null : dirname(root), images:files, dirs}
}

async function readStatus(dir) {
  try { return JSON.parse(await readFile(join(dir, 'status.json'), 'utf8')) } catch { return null }
}

async function snapshot() {
  if (!current) return {job:null}
  const status = await readStatus(current.dir) ?? {stage:'starting'}
  if (current.exit !== undefined && !['done', 'error'].includes(status.stage)) {
    status.stage = current.cancelled ? 'cancelled' : 'error'
    status.message = current.cancelled ? '中止しました' : status.message || '処理が異常終了しました（コード ' + current.exit + '）'
  }
  return {job:{id:current.id, request:current.request, running:current.exit === undefined, ...status}}
}

async function start(body) {
  if (current && current.exit === undefined) throw Error('別の学習が実行中です')
  const p = await plan(body)
  if (p.exists) throw Error('同じ名前の LoRA があります。出力名を変えてください: ' + p.comfyName)
  const {images} = await scan(p.folder)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-' + p.name
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {folder:p.folder, images, trigger:p.trigger, overwrite:p.overwrite, purpose:body.purpose,
                   instruction:CAPTION_INSTRUCTIONS[body.purpose], output:p.output, steps:p.settings.steps,
                   train_args:p.settings.args, comfyName:p.comfyName, settings:p.settings}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  const child = spawn(PYTHON, ['-I', PIPELINE, dir], {cwd:ROOT, detached:true, stdio:['ignore', 'ignore', 'pipe']})
  const job = {id, dir, request, process:child, exit:undefined, cancelled:false}
  let stderr = ''
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000) })
  child.on('exit', code => {
    job.exit = code ?? 1
    if (code && stderr) writeFile(join(dir, 'pipeline.err'), stderr).catch(() => {})
  })
  child.on('error', error => { job.exit = 1; writeFile(join(dir, 'pipeline.err'), String(error)).catch(() => {}) })
  current = job
  return snapshot()
}

function cancel() {
  if (!current || current.exit !== undefined) return snapshot()
  current.cancelled = true
  // The pipeline, llama-server and the trainer share this process group.
  try { process.kill(-current.process.pid, 'SIGTERM') } catch {}
  const job = current
  setTimeout(() => { if (job.exit === undefined) try { process.kill(-job.process.pid, 'SIGKILL') } catch {} }, 8000)
  return snapshot()
}

export function registerJobApi(web) {
  web.effect(() => web.connection.fetch.register({
    path:'/api/darask.animaLora', methods:['GET', 'POST'], requestBody:'buffered',
    async fetch(request) {
      const reply = (data, status = 200) => Response.json(data, {status, headers:{'cache-control':'no-store'}})
      try {
        if (request.method === 'GET') return reply({...(await snapshot()), purposes:Object.fromEntries(
          Object.entries(PURPOSES).map(([key, value]) => [key, value.label]))})
        if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') return reply({error:'JSON が必要です'}, 415)
        const body = await request.json()
        if (body.action === 'list') return reply(await list(body.path))
        if (body.action === 'plan') return reply(await plan(body))
        if (body.action === 'start') return reply(await start(body))
        if (body.action === 'cancel') return reply(await cancel())
        return reply({error:'不明な操作です'}, 400)
      } catch (error) {
        return reply({error:error.message}, 400)
      }
    },
  }), 'Anima LoRA caption and training jobs')
  // Stop a running job when the plugin unloads (service restart).
  web.effect(() => () => { if (current && current.exit === undefined) cancel() }, 'Stop Anima LoRA job on unload')
}
