// Owner-only API behind the "Anima LoRA" and "Anima 生成" sidebar panels: scan a folder, recommend
// settings, then caption with Gemma and train natively; or generate images natively (text-to-image,
// bit-exact with ComfyUI). Training and generation share one queue, so they take the GPU in turn;
// the queue survives service restarts (queue.json).
import {spawn} from 'node:child_process'
import {randomInt} from 'node:crypto'
import {existsSync} from 'node:fs'
import {mkdir, open, readdir, readFile, realpath, rename, stat, writeFile} from 'node:fs/promises'
import {homedir} from 'node:os'
import {basename, dirname, extname, isAbsolute, join, relative, sep} from 'node:path'
import {fileURLToPath} from 'node:url'
import {CAPTION_INSTRUCTIONS, PURPOSES, recommend} from './presets.mjs'

const ROOT = fileURLToPath(new URL('../..', import.meta.url))
const PIPELINE = fileURLToPath(new URL('./pipeline.py', import.meta.url))
const PYTHON = process.env.DSH_ANIMA_PYTHON || join(ROOT, '.runtime/lora-venv/bin/python')
const JOBS = join(ROOT, '.runtime/anima-native/jobs')
const OUTPUT_DIR = join(ROOT, 'plugins/dsh-comfyui/loras/trained')
const GENERATE = fileURLToPath(new URL('./native/generate/generate.py', import.meta.url))
const GENERATE_PYTHON = process.env.DSH_ANIMA_GENERATE_PYTHON || join(ROOT, '.runtime/comfy-venv/bin/python')
const LORA_DIR = join(ROOT, 'plugins/dsh-comfyui/models/loras')
const GENERATED = join(ROOT, '.runtime/anima-native/generated')
// MiniMax H3 text-to-video (plugins/dsh-minimax-h3/native, same Python as Anima generation)
const H3 = join(ROOT, 'plugins/dsh-minimax-h3/native/generate.py')
const H3_LORA = {name:'H3_VBVR_Pro-V2', path:join(ROOT, 'plugins/dsh-comfyui/loras/H3_VBVR_Pro-V2.safetensors'), strength:1}
const H3_GENERATED = join(ROOT, '.runtime/h3-native/generated')
const PH3 = join(ROOT, 'plugins/dsh-minimax-h3/native/ph3.py')
const IMAGES = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'])


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
  const queued = jobs.some(job => ['queued', 'running'].includes(job.state) && job.request.output === output)
  return {folder:root, images:images.length, captioned, trigger, name, output, comfyName:'trained/' + name + '.safetensors',
          exists:existsSync(output) || queued, queued, settings, overwrite:body.overwrite !== false}
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

/** Anima DiT LoRAs under models/loras (ComfyUI names); other formats (H3, adapters) are left out by their keys. */
async function listLoras() {
  const found = []
  async function walk(dir) {
    for (const entry of await readdir(dir, {withFileTypes:true}).catch(() => [])) {
      const path = join(dir, entry.name)
      if (entry.isDirectory()) await walk(path)
      else if (entry.isFile() && entry.name.endsWith('.safetensors') && await isAnimaLora(path))
        found.push({name:relative(LORA_DIR, path).split(sep).join('/'), path})
    }
  }
  await walk(LORA_DIR)
  return found.sort((a, b) => a.name.localeCompare(b.name))
}

// Every key must name a module of the Anima (Cosmos Predict2) DiT; H3 LoRAs also use blocks.N.attn/mlp.fc*.
const ANIMA_LORA_KEY = /^diffusion_model\.(blocks\.\d+\.((self|cross)_attn\.(q|k|v|output)_proj|mlp\.layer[12]|adaln_modulation_(self_attn|cross_attn|mlp)\.[12])|final_layer\.(linear|adaln_modulation\.[12])|t_embedder\.1\.linear_[12]|x_embedder\.proj\.1)\.(lora_A\.weight|lora_B\.weight|alpha)$/

async function isAnimaLora(path) {
  let file
  try {
    file = await open(path)
    const size = Buffer.alloc(8)
    await file.read(size, 0, 8, 0)
    const length = Number(size.readBigUInt64LE())
    if (length > 64 << 20) return false
    const header = Buffer.alloc(length)
    await file.read(header, 0, length, 8)
    const keys = Object.keys(JSON.parse(header.toString('utf8'))).filter(key => key !== '__metadata__')
    return keys.length > 0 && keys.every(key => ANIMA_LORA_KEY.test(key))
  } catch { return false } finally { await file?.close() }
}

function number(value, label, min, max, integer = true) {
  const n = Number(value)
  if (!Number.isFinite(n) || n < min || n > max || (integer && !Number.isInteger(n)))
    throw Error(label + 'は ' + min + '〜' + max + (integer ? ' の整数' : '') + 'で指定してください')
  return n
}

async function enqueueGenerate(body) {
  await load()
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''
  const negative = typeof body.negative === 'string' ? body.negative.trim() : ''
  if (!prompt) throw Error('プロンプトを入力してください')
  if (prompt.length > 4000 || negative.length > 4000) throw Error('プロンプトは 4000 文字以内にしてください')
  const width = number(body.width, '幅', 256, 2048), height = number(body.height, '高さ', 256, 2048)
  if (width % 16 || height % 16) throw Error('幅と高さは 16 の倍数にしてください')
  const available = new Map((await listLoras()).map(item => [item.name, item.path]))
  const loras = (Array.isArray(body.loras) ? body.loras : []).slice(0, 4).map(item => {
    if (!available.has(item?.name)) throw Error('LoRA が見つかりません: ' + item?.name)
    return {name:item.name, path:available.get(item.name), strength:number(item.strength, 'LoRA の強さ', -2, 2, false)}
  })
  const seed = body.seed === null || body.seed === undefined || body.seed === '' ? randomInt(0, 2 ** 32)
    : number(body.seed, 'シード', 0, Number.MAX_SAFE_INTEGER)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-gen'
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {kind:'generate', prompt, negative, seed, count:number(body.count, '枚数', 1, 16),
                   steps:number(body.steps, 'step 数', 1, 100), cfg:number(body.cfg, 'CFG', 1, 20, false),
                   width, height, loras, fp8:body.fp8 !== false, output_dir:join(GENERATED, id)}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return snapshot()
}

async function enqueueVideo(body) {
  await load()
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''
  if (!prompt) throw Error('プロンプトを入力してください')
  if (prompt.length > 4000) throw Error('プロンプトは 4000 文字以内にしてください')
  const width = number(body.width, '幅', 256, 1344), height = number(body.height, '高さ', 256, 1344)
  if (width % 32 || height % 32) throw Error('幅と高さは 32 の倍数にしてください')
  if (width * height > 1344 * 768) throw Error('大きさは 1344×768 の面積までにしてください')
  const seed = body.seed === null || body.seed === undefined || body.seed === '' ? randomInt(0, 2 ** 32)
    : number(body.seed, 'シード', 0, Number.MAX_SAFE_INTEGER)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-h3'
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {kind:'h3', prompt, seed, width, height, length:number(body.length, 'フレーム数', 5, 362),
                   loras:[H3_LORA], output_dir:join(H3_GENERATED, id)}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return snapshot()
}

async function enqueueImage(body) {
  await load()
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''
  if (!prompt) throw Error('プロンプトを入力してください')
  if (prompt.length > 4000) throw Error('プロンプトは 4000 文字以内にしてください')
  const width = number(body.width, '幅', 256, 4096), height = number(body.height, '高さ', 256, 4096)
  if (width % 32 || height % 32) throw Error('幅と高さは 32 の倍数にしてください')
  if (width * height > 2912 * 2912) throw Error('大きさは 2912×2912 の面積までにしてください')
  const seed = body.seed === null || body.seed === undefined || body.seed === '' ? randomInt(0, 2 ** 32)
    : number(body.seed, 'シード', 0, Number.MAX_SAFE_INTEGER)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-ph3'
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {kind:'ph3', prompt, seed, count:number(body.count, '枚数', 1, 16), width, height,
                   output_dir:join(H3_GENERATED, id)}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return snapshot()
}

/** A generated file of a known job: GET ?image=<job id>/<NNN-seed.png | seed.mp4> */
async function image(name) {
  const [id, file, ...rest] = String(name).split('/')
  const job = jobs.find(job => job.id === id)
  const pattern = {generate:/^\d{3}-\d+\.png$/, ph3:/^\d{3}-\d+\.png$/, h3:/^\d+\.mp4$/}[job?.request.kind]
  if (!pattern || rest.length || !pattern.test(file)) return null
  const data = await readFile(join(job.request.output_dir, file)).catch(() => null)
  return data && {data, type:job.request.kind === 'h3' ? 'video/mp4' : 'image/png'}
}

const QUEUE = join(JOBS, 'queue.json')
const KEEP_FINISHED = 20
let jobs = []  // [{id, dir, request, state: queued|running|done|error|cancelled, process?, message?}]
let loaded = null
let stopping = false

async function load() {
  loaded ??= (async () => {
    try { jobs = JSON.parse(await readFile(QUEUE, 'utf8')).map(job => ({id:job.id, dir:job.dir, request:job.request,
      // A job cut off by a service restart starts over (its LoRA was not written yet).
      state:job.state === 'running' ? 'queued' : job.state,
      message:job.state === 'running' ? 'DSH の再起動で中断したため再実行します' : job.message})) } catch { jobs = [] }
    await save()
    pump()
  })()
  return loaded
}

async function save() {
  const finished = jobs.filter(job => !['queued', 'running'].includes(job.state))
  const drop = new Set(finished.slice(0, Math.max(0, finished.length - KEEP_FINISHED)))
  jobs = jobs.filter(job => !drop.has(job))
  await mkdir(JOBS, {recursive:true})
  const body = JSON.stringify(jobs.map(({id, dir, request, state, message}) => ({id, dir, request, state, message})), null, 1)
  await writeFile(QUEUE + '.tmp', body)
  await rename(QUEUE + '.tmp', QUEUE)
}

async function readStatus(dir) {
  try { return JSON.parse(await readFile(join(dir, 'status.json'), 'utf8')) } catch { return null }
}

function pump() {
  if (stopping || jobs.some(job => job.state === 'running')) return
  const job = jobs.find(job => job.state === 'queued')
  if (!job) return
  job.state = 'running'
  const script = {generate:GENERATE, h3:H3, ph3:PH3}[job.request.kind]
  // generate.py builds its C++ ops on first use; ninja comes from the trainer's venv.
  const env = script ? {...process.env, PATH:join(ROOT, '.runtime/lora-venv/bin') + ':' + process.env.PATH} : process.env
  const child = spawn(script ? GENERATE_PYTHON : PYTHON, ['-I', script ?? PIPELINE, job.dir],
                      {cwd:ROOT, detached:true, env, stdio:['ignore', 'ignore', 'pipe']})
  job.process = child
  let stderr = ''
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000) })
  const finish = async code => {
    // While the plugin unloads, leave the job as running in queue.json so it is re-queued on the next load.
    if (job.state !== 'running' || stopping) return
    const status = await readStatus(job.dir)
    job.state = job.cancelled ? 'cancelled' : status?.stage === 'done' && !code ? 'done' : 'error'
    job.message = job.cancelled ? '中止しました' : status?.stage === 'error' ? status.message
      : code ? '処理が異常終了しました（コード ' + code + '）' : undefined
    if (code && stderr) await writeFile(join(job.dir, 'pipeline.err'), stderr).catch(() => {})
    delete job.process
    await save().catch(() => {})
    pump()
  }
  child.on('exit', code => finish(code ?? 1))
  child.on('error', error => { stderr += String(error); finish(1) })
  save().catch(() => {})
}

async function snapshot() {
  await load()
  const position = new Map(jobs.filter(job => job.state === 'queued').map((job, index) => [job.id, index + 1]))
  const views = await Promise.all(jobs.map(async job => {
    const status = job.state === 'queued' ? {} : await readStatus(job.dir) ?? {}
    const view = {...status, id:job.id, request:job.request, state:job.state, running:job.state === 'running',
                  queued:job.state === 'queued', position:position.get(job.id)}
    view.stage = job.state === 'running' ? status.stage || 'starting' : job.state === 'error' ? 'error' : job.state
    if (job.message) view.message = job.message
    return view
  }))
  // Running first, then the queue in order, then finished jobs newest first.
  const rank = view => view.running ? 0 : view.queued ? 1 : 2
  views.sort((a, b) => rank(a) - rank(b) || (rank(a) === 2 ? b.id.localeCompare(a.id) : 0))
  return {jobs:views}
}

async function enqueue(body) {
  await load()
  const p = await plan(body)
  if (p.exists) throw Error(p.queued ? 'キューに同じ出力名の LoRA があります: ' + p.comfyName
                                     : '同じ名前の LoRA があります。出力名を変えてください: ' + p.comfyName)
  const {images} = await scan(p.folder)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-' + p.name
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {folder:p.folder, images, trigger:p.trigger, overwrite:p.overwrite, purpose:body.purpose,
                   instruction:CAPTION_INSTRUCTIONS[body.purpose], output:p.output, steps:p.settings.steps,
                   train_args:p.settings.args, comfyName:p.comfyName, settings:p.settings}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return snapshot()
}

async function cancel(id) {
  await load()
  const job = jobs.find(job => job.id === id)
  if (!job) throw Error('ジョブが見つかりません')
  if (job.state === 'queued') {
    job.state = 'cancelled'
    job.message = '開始前に取り消しました'
    await save()
  } else if (job.state === 'running') {
    job.cancelled = true
    // The pipeline, llama-server and the trainer share this process group.
    const pid = job.process.pid
    try { process.kill(-pid, 'SIGTERM') } catch {}
    setTimeout(() => { if (job.state === 'running') try { process.kill(-pid, 'SIGKILL') } catch {} }, 8000)
  }
  return snapshot()
}

async function clearFinished() {
  await load()
  jobs = jobs.filter(job => ['queued', 'running'].includes(job.state))
  await save()
  return snapshot()
}

export function registerJobApi(web) {
  web.effect(() => web.connection.fetch.register({
    path:'/api/darask.animaLora', methods:['GET', 'POST'], requestBody:'buffered',
    async fetch(request) {
      const reply = (data, status = 200) => Response.json(data, {status, headers:{'cache-control':'no-store'}})
      try {
        if (request.method === 'GET') {
          const name = new URL(request.url, 'http://local').searchParams.get('image')
          if (name !== null) {
            await load()
            const file = await image(name)
            return file ? new Response(file.data, {headers:{'content-type':file.type, 'cache-control':'private, max-age=86400'}})
              : reply({error:'ファイルが見つかりません'}, 404)
          }
        }
        if (request.method === 'GET') return reply({...(await snapshot()), purposes:Object.fromEntries(
          Object.entries(PURPOSES).map(([key, value]) => [key, value.label]))})
        if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') return reply({error:'JSON が必要です'}, 415)
        const body = await request.json()
        if (body.action === 'list') return reply(await list(body.path))
        if (body.action === 'plan') { await load(); return reply(await plan(body)) }
        if (body.action === 'start') return reply(await enqueue(body))
        if (body.action === 'generate') return reply(await enqueueGenerate(body))
        if (body.action === 'video') return reply(await enqueueVideo(body))
        if (body.action === 'image') return reply(await enqueueImage(body))
        if (body.action === 'loras') return reply({loras:(await listLoras()).map(item => item.name)})
        if (body.action === 'cancel') return reply(await cancel(body.id))
        if (body.action === 'clear') return reply(await clearFinished())
        return reply({error:'不明な操作です'}, 400)
      } catch (error) {
        return reply({error:error.message}, 400)
      }
    },
  }), 'Anima LoRA caption and training queue')
  web.effect(() => { stopping = false; load(); return () => {
    stopping = true
    // Stop the running job when the plugin unloads; it is recorded as interrupted on the next load.
    for (const job of jobs) if (job.state === 'running') try { process.kill(-job.process.pid, 'SIGTERM') } catch {}
  } }, 'Anima LoRA queue')
}
