// Owner-only API behind the "Anima LoRA" and "Anima 生成" sidebar panels: scan a folder, recommend
// settings, then caption with Gemma and train natively; or generate images natively (text-to-image,
// bit-exact with ComfyUI). Training and generation share one queue, so they take the GPU in turn;
// the queue survives service restarts (queue.json).
import {spawn} from 'node:child_process'
import {randomBytes, randomInt} from 'node:crypto'
import {existsSync} from 'node:fs'
import {mkdir, open, readdir, readFile, realpath, rename, rm, stat, writeFile} from 'node:fs/promises'
import {homedir} from 'node:os'
import {basename, dirname, extname, isAbsolute, join, relative, sep} from 'node:path'
import {fileURLToPath} from 'node:url'
import {CAPTION_INSTRUCTIONS, MODELS, PURPOSES, recommend} from './presets.mjs'

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
const VIDEOS = new Set(['.mp4', '.mov', '.webm', '.mkv', '.avi'])
// Pictures sent from the browser (local or remote) or a chat tool: keyframes, references, edit inputs.
const UPLOADS = join(ROOT, '.runtime/anima-native/uploads')
// Training folders built from uploaded images and captions (remote LoRA training).
const DATASETS = join(ROOT, '.runtime/anima-native/datasets')
const PICTURES = new Set(['.png', '.jpg', '.jpeg', '.webp'])
const MAX_UPLOAD = 20 << 20
const UPLOAD_DAYS = 14

/** Stores one picture and returns its token (the file name under UPLOADS). */
export async function storePicture(data, ext) {
  ext = String(ext).toLowerCase()
  if (!PICTURES.has(ext)) throw Error('画像（png・jpg・webp）を選んでください')
  if (!data.length || data.length > MAX_UPLOAD) throw Error('画像は 20 MB までです')
  const token = randomBytes(16).toString('hex') + ext
  await mkdir(UPLOADS, {recursive:true})
  await writeFile(join(UPLOADS, token), data, {flag:'wx'})
  return token
}

async function upload(body) {
  return {token:await storePicture(Buffer.from(String(body.data || ''), 'base64'), extname(String(body.name || '')))}
}

function picture(token, label) {
  if (typeof token !== 'string' || !/^[0-9a-f]{32}\.(png|jpe?g|webp)$/.test(token) || !existsSync(join(UPLOADS, token)))
    throw Error(label + 'の画像が見つかりません。選び直してください')
  return join(UPLOADS, token)
}

function pictures(tokens, label, min, max) {
  const list = Array.isArray(tokens) ? tokens : []
  if (list.length < min || list.length > max) throw Error(label + 'は ' + min + '〜' + max + ' 枚にしてください')
  return list.map((token, index) => picture(token, label + ' ' + (index + 1)))
}

/** Adds one uploaded file (image or same-stem .txt caption) to a training folder; never overwrites. */
async function datasetFile(body) {
  const dataset = String(body.dataset || '')
  if (!/^[\w.@-]{1,80}$/.test(dataset) || dataset.startsWith('.')) throw Error('フォルダ名は英数字と . _ - @ で 80 文字以内にしてください')
  const name = basename(String(body.name || ''))
  const ext = extname(name).toLowerCase()
  if (!name || name.startsWith('.') || /[\u0000-\u001f/\\]/.test(name) || !(IMAGES.has(ext) || ext === '.txt'))
    throw Error('画像と同じ名前の .txt だけを追加できます: ' + name)
  const data = Buffer.from(String(body.data || ''), 'base64')
  if (data.length > MAX_UPLOAD) throw Error('1 ファイル 20 MB までです: ' + name)
  const folder = join(DATASETS, dataset)
  await mkdir(folder, {recursive:true})
  await writeFile(join(folder, name), data, {flag:'wx'}).catch(error => {
    throw error.code === 'EEXIST' ? Error('同じ名前のファイルがあります（上書きしません）: ' + name) : error
  })
  return {folder}
}

async function pruneUploads() {
  const limit = Date.now() - UPLOAD_DAYS * 86400000
  const keep = new Set(jobs.filter(job => ['queued', 'running'].includes(job.state))
    .flatMap(job => [job.request.first_frame, job.request.last_frame, job.request.image, job.request.mask,
                     ...(job.request.references || []), ...(job.request.images || [])]))
  for (const name of await readdir(UPLOADS).catch(() => [])) {
    const path = join(UPLOADS, name)
    if (!keep.has(path) && (await stat(path).catch(() => null))?.mtimeMs < limit) await rm(path, {force:true})
  }
}


async function scan(folder, model = 'anima') {
  if (typeof folder !== 'string' || !isAbsolute(folder.trim())) throw Error('フォルダを絶対パスで指定してください')
  const root = await realpath(folder.trim())
  if (!(await stat(root)).isDirectory()) throw Error('フォルダを指定してください')
  const images = [], captioned = []
  const videos = MODELS[model]?.videos
  for (const entry of (await readdir(root, {withFileTypes:true})).sort((a, b) => a.name.localeCompare(b.name))) {
    const ext = extname(entry.name).toLowerCase()
    if (!entry.isFile() || !(IMAGES.has(ext) || videos && VIDEOS.has(ext))) continue
    const path = join(root, entry.name)
    images.push(path)
    const caption = path.slice(0, -extname(path).length) + '.txt'
    if (existsSync(caption) && (await readFile(caption, 'utf8')).trim()) captioned.push(path)
  }
  if (!images.length) throw Error(videos ? 'フォルダに画像・動画がありません' : 'フォルダに画像（png・jpg・webp など）がありません')
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
  const model = MODELS[body.model] ? body.model : 'anima'
  const {root, images, captioned} = await scan(body.folder, model)
  const trigger = checkTrigger(body.trigger)
  const settings = recommend(body.purpose, images.length, model)
  const name = outputName(body.name, trigger, root, (model === 'anima' ? '' : model + '-') + body.purpose)
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

/** LoRAs under models/loras (ComfyUI names) usable with `model` ('anima', 'h3' or 'ph3'), told apart by their keys and the
 *  trainer's dsh_model metadata. The VBVR LoRA H3 always applies is left out. */
async function listLoras(model = 'anima') {
  const found = []
  async function walk(dir) {
    for (const entry of await readdir(dir, {withFileTypes:true}).catch(() => [])) {
      const path = join(dir, entry.name)
      if (entry.isDirectory()) await walk(path)
      else if (entry.isFile() && entry.name.endsWith('.safetensors') && entry.name !== basename(H3_LORA.path)) {
        const kind = await loraKind(path)
        if (kind === model || kind === 'minimax' && model !== 'anima')
          found.push({name:relative(LORA_DIR, path).split(sep).join('/'), path})
      }
    }
  }
  await walk(LORA_DIR)
  return found.sort((a, b) => a.name.localeCompare(b.name))
}

// Every key must name a module of the Anima (Cosmos Predict2) DiT; H3 LoRAs also use blocks.N.attn/mlp.fc*.
const ANIMA_LORA_KEY = /^diffusion_model\.(blocks\.\d+\.((self|cross)_attn\.(q|k|v|output)_proj|mlp\.layer[12]|adaln_modulation_(self_attn|cross_attn|mlp)\.[12])|final_layer\.(linear|adaln_modulation\.[12])|t_embedder\.1\.linear_[12]|x_embedder\.proj\.1)\.(lora_A\.weight|lora_B\.weight|alpha)$/
// MiniMax H3 / P-H3 (same DiT): diffusion-pipe / PEFT or kohya keys on the blocks and the token refiner.
const MINIMAX_LORA_KEY = /^(diffusion_model\.(token_refiner\.)?blocks\.\d+\.(attn\.(qkv|out)_proj|mlp\.fc[12])\.lora_[AB]\.weight|lora_unet_(token_refiner_)?blocks_\d+_(attn_(qkv|out)_proj|mlp_fc[12])\.(lora_up\.weight|lora_down\.weight|alpha))$/

/** 'anima', 'h3' / 'ph3' (trained here: metadata dsh_model), 'minimax' (either of those) or null. */
async function loraKind(path) {
  let file
  try {
    file = await open(path)
    const size = Buffer.alloc(8)
    await file.read(size, 0, 8, 0)
    const length = Number(size.readBigUInt64LE())
    if (length > 64 << 20) return null
    const header = Buffer.alloc(length)
    await file.read(header, 0, length, 8)
    const parsed = JSON.parse(header.toString('utf8'))
    const keys = Object.keys(parsed).filter(key => key !== '__metadata__')
    if (!keys.length) return null
    if (keys.every(key => ANIMA_LORA_KEY.test(key))) return 'anima'
    if (keys.every(key => MINIMAX_LORA_KEY.test(key))) return ['h3', 'ph3'].includes(parsed.__metadata__?.dsh_model) ? parsed.__metadata__.dsh_model : 'minimax'
    return null
  } catch { return null } finally { await file?.close() }
}

/** body.loras [{name, strength}] -> [{name, path, strength}] for `model` (up to 4). */
async function chosenLoras(body, model) {
  const available = new Map((await listLoras(model)).map(item => [item.name, item.path]))
  return (Array.isArray(body.loras) ? body.loras : []).slice(0, 4).map(item => {
    if (!available.has(item?.name)) throw Error('LoRA が見つかりません: ' + item?.name)
    return {name:item.name, path:available.get(item.name), strength:number(item.strength, 'LoRA の強さ', -2, 2, false)}
  })
}

function number(value, label, min, max, integer = true) {
  const n = Number(value)
  if (!Number.isFinite(n) || n < min || n > max || (integer && !Number.isInteger(n)))
    throw Error(label + 'は ' + min + '〜' + max + (integer ? ' の整数' : '') + 'で指定してください')
  return n
}

export async function enqueueGenerate(body) {
  await load()
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''
  const negative = typeof body.negative === 'string' ? body.negative.trim() : ''
  if (!prompt) throw Error('プロンプトを入力してください')
  if (prompt.length > 4000 || negative.length > 4000) throw Error('プロンプトは 4000 文字以内にしてください')
  // Editing (img2img / redraw inside a mask): the picture to edit and, optionally, the mask (white = redraw).
  // 0 x 0 keeps the picture's aspect at about 1.2 MP (generate.py).
  const edit = body.image ? {image:picture(body.image, '編集する画像'), denoise:number(body.denoise ?? 0.6, '描き直しの強さ', 0.05, 1, false),
    ...(body.mask ? {mask:picture(body.mask, 'マスク'), mask_grow:number(body.mask_grow ?? 8, 'マスクの広げ', 0, 64),
                     mask_feather:number(body.mask_feather ?? 12, 'マスクのぼかし', 0, 64)} : {})} : null
  const auto = edit && !Number(body.width) && !Number(body.height)
  const width = auto ? 0 : number(body.width, '幅', 256, 2048), height = auto ? 0 : number(body.height, '高さ', 256, 2048)
  if (width % 16 || height % 16) throw Error('幅と高さは 16 の倍数にしてください')
  const loras = await chosenLoras(body, 'anima')
  const seed = body.seed === null || body.seed === undefined || body.seed === '' ? randomInt(0, 2 ** 32)
    : number(body.seed, 'シード', 0, Number.MAX_SAFE_INTEGER)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-gen'
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {kind:'generate', prompt, negative, seed, count:number(body.count, '枚数', 1, 16),
                   steps:number(body.steps, 'step 数', 1, 100), cfg:number(body.cfg, 'CFG', 1, 20, false),
                   width, height, loras, fp8:body.fp8 !== false, ...(edit || {}), output_dir:join(GENERATED, id)}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return {id}
}

export async function enqueueVideo(body) {
  await load()
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''
  if (!prompt) throw Error('プロンプトを入力してください')
  if (prompt.length > 4000) throw Error('プロンプトは 4000 文字以内にしてください')
  const width = number(body.width, '幅', 256, 1344), height = number(body.height, '高さ', 256, 1344)
  if (width % 32 || height % 32) throw Error('幅と高さは 32 の倍数にしてください')
  if (width * height > 1344 * 768) throw Error('大きさは 1344×768 の面積までにしてください')
  const mode = body.mode || 't2v'
  const inputs = {}
  if (mode === 'i2v') {
    if (!body.first && !body.last) throw Error('最初か最後のフレームの画像を選んでください')
    if (body.first) inputs.first_frame = picture(body.first, '最初のフレーム')
    if (body.last) inputs.last_frame = picture(body.last, '最後のフレーム')
  } else if (mode === 'ref') {
    inputs.references = pictures(body.references, '参照画像', 1, 9)
  } else if (mode !== 't2v') throw Error('不明なモードです')
  const seed = body.seed === null || body.seed === undefined || body.seed === '' ? randomInt(0, 2 ** 32)
    : number(body.seed, 'シード', 0, Number.MAX_SAFE_INTEGER)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-h3'
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const loras = [H3_LORA, ...await chosenLoras(body, 'h3')]  // VBVR first, then the user's (merged in this order)
  const request = {kind:'h3', mode, prompt, seed, width, height, length:number(body.length, 'フレーム数', 5, 1440),
                   loras, ...inputs, output_dir:join(H3_GENERATED, id)}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return {id}
}

export async function enqueueImage(body) {
  await load()
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''
  if (!prompt) throw Error('プロンプトを入力してください')
  if (prompt.length > 4000) throw Error('プロンプトは 4000 文字以内にしてください')
  const mode = body.mode || 't2i'
  if (!['t2i', 'edit', 'reference'].includes(mode)) throw Error('不明なモードです')
  const images = mode === 't2i' ? [] : pictures(body.images, mode === 'edit' ? '編集する画像と参照画像' : '参照画像', 1, 9)
  // edit: 0 x 0 keeps the size of the picture to edit
  const auto = mode === 'edit' && !Number(body.width) && !Number(body.height)
  const width = auto ? 0 : number(body.width, '幅', 256, 4096), height = auto ? 0 : number(body.height, '高さ', 256, 4096)
  if (width % 32 || height % 32) throw Error('幅と高さは 32 の倍数にしてください')
  if (width * height > 2912 * 2912) throw Error('大きさは 2912×2912 の面積までにしてください')
  const seed = body.seed === null || body.seed === undefined || body.seed === '' ? randomInt(0, 2 ** 32)
    : number(body.seed, 'シード', 0, Number.MAX_SAFE_INTEGER)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-ph3'
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const loras = await chosenLoras(body, 'ph3')
  const request = {kind:'ph3', mode, prompt, seed, count:number(body.count, '枚数', 1, 16), width, height,
                   ...(images.length ? {images} : {}), ...(loras.length ? {loras} : {}), output_dir:join(H3_GENERATED, id)}
  await writeFile(join(dir, 'request.json'), JSON.stringify(request, null, 1))
  jobs.push({id, dir, request, state:'queued'})
  await save()
  pump()
  return {id}
}

/** A generated file of a known job (GET ?image=<job id>/<NNN-seed.png | seed.mp4>) or an uploaded picture
 *  (GET ?upload=<token>): {path, type} */
function mediaPath(params) {
  const upload = params.get('upload')
  if (upload !== null) {
    if (!/^[0-9a-f]{32}\.(png|jpe?g|webp)$/.test(upload)) return null
    const ext = extname(upload).slice(1)
    return {path:join(UPLOADS, upload), type:'image/' + (ext === 'jpg' ? 'jpeg' : ext)}
  }
  const [id, file, ...rest] = String(params.get('image')).split('/')
  const job = jobs.find(job => job.id === id)
  const pattern = {generate:/^\d{3}-\d+\.png$/, ph3:/^\d{3}-\d+\.png$/, h3:/^\d+\.mp4$/}[job?.request.kind]
  if (!pattern || rest.length || !pattern.test(file)) return null
  return {path:join(job.request.output_dir, file), type:job.request.kind === 'h3' ? 'video/mp4' : 'image/png'}
}

/** Serves a media file; videos answer Range requests (206) so phones can play and seek them. */
async function media(request, params) {
  const found = mediaPath(params)
  const data = found && await readFile(found.path).catch(() => null)
  if (!data) return Response.json({error:'ファイルが見つかりません'}, {status:404, headers:{'cache-control':'no-store'}})
  const headers = {'content-type':found.type, 'cache-control':'private, max-age=86400', 'accept-ranges':'bytes'}
  const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.get('range') || '')
  if (!range || (!range[1] && !range[2])) return new Response(data, {headers})
  let start = range[1] ? Number(range[1]) : Math.max(0, data.length - Number(range[2]))
  let end = range[1] && range[2] ? Math.min(Number(range[2]), data.length - 1) : data.length - 1
  if (start > end || start >= data.length)
    return new Response(null, {status:416, headers:{...headers, 'content-range':'bytes */' + data.length}})
  return new Response(data.subarray(start, end + 1), {status:206,
    headers:{...headers, 'content-range':'bytes ' + start + '-' + end + '/' + data.length}})
}

/** The job's current view (for chat tools waiting on their own job). */
export async function jobView(id) {
  return (await snapshot()).jobs.find(job => job.id === id) ?? null
}

/** The output files of a finished job, as absolute paths. */
export function jobFiles(view) {
  return (view?.images || []).map(item => join(view.request.output_dir, item.file))
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
    pruneUploads().catch(() => {})
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
  const {images} = await scan(p.folder, p.settings.model)
  const id = new Date().toISOString().replace(/[:.]/g, '-') + '-' + p.name
  const dir = join(JOBS, id)
  await mkdir(dir, {recursive:true})
  const request = {model:p.settings.model, folder:p.folder, images, trigger:p.trigger, overwrite:p.overwrite, purpose:body.purpose,
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

export {cancel as cancelJob}

export async function listLoraNames(model = 'anima') {
  return (await listLoras(model)).map(item => item.name)
}

export function registerJobApi(web) {
  web.effect(() => web.connection.fetch.register({
    path:'/api/darask.animaLora', methods:['GET', 'POST'], requestBody:'buffered',
    async fetch(request) {
      const reply = (data, status = 200) => Response.json(data, {status, headers:{'cache-control':'no-store'}})
      try {
        if (request.method === 'GET') {
          const params = new URL(request.url, 'http://local').searchParams
          if (params.has('image') || params.has('upload')) {
            await load()
            return await media(request, params)
          }
        }
        if (request.method === 'GET') return reply({...(await snapshot()), purposes:Object.fromEntries(
          Object.entries(PURPOSES).map(([key, value]) => [key, value.label]))})
        if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') return reply({error:'JSON が必要です'}, 415)
        const body = await request.json()
        if (body.action === 'list') return reply(await list(body.path))
        if (body.action === 'plan') { await load(); return reply(await plan(body)) }
        if (body.action === 'start') return reply(await enqueue(body))
        if (body.action === 'generate') return reply({...await enqueueGenerate(body), ...await snapshot()})
        if (body.action === 'video') return reply({...await enqueueVideo(body), ...await snapshot()})
        if (body.action === 'image') return reply({...await enqueueImage(body), ...await snapshot()})
        if (body.action === 'upload') return reply(await upload(body))
        if (body.action === 'dataset') return reply(await datasetFile(body))
        if (body.action === 'loras') return reply({loras:await listLoraNames(['h3', 'ph3'].includes(body.model) ? body.model : 'anima')})
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
