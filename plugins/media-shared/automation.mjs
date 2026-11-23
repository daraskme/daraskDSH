// External automation API at /automation/v1: off-box agents (Muse) submit and poll media jobs with
// a Bearer key instead of a DSH web session. It is a webServer prefix route inside this Harness
// process and calls the same enqueue*/job functions as the sidebar panels and chat tools, so every
// automation job takes its turn in the one GPU job queue (queue.json) — nothing runs in parallel.
// Cloudflare Access is bypassed for this exact path by a dashboard Bypass policy; the Bearer key
// check below replaces it (hash compare only — the raw key is never stored or logged).
import {createHash, timingSafeEqual} from 'node:crypto'
import {readFile} from 'node:fs/promises'
import {basename, extname} from 'node:path'
import {cancelJob, enqueueGenerate, enqueueImage, enqueueVideo, jobFiles, jobView, jobsSnapshot, listLoraNames, storePicture} from './jobs.mjs'
import {size} from './tool-support.mjs'

/** enqueue(), adding the usable LoRA names to a LoRA error — same UX as the chat tools. */
async function enqueueWithLoras(model, deps, enqueue) {
  try { return await enqueue() } catch (error) {
    if (!/LoRA/.test(error.message)) throw error
    throw Error(error.message + '（使える LoRA: ' + ((await deps.listLoraNames(model)).join(', ') || 'なし') + '）')
  }
}

const PREFIX = '/automation/v1'
const KINDS = ['image', 'edit', 'video']
const MAX_BODY = 320 << 20 // 8 data-URI images at the 20 MB upload cap, base64-expanded
const TYPES = {'.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.webp':'image/webp', '.mp4':'video/mp4'}

const real = {enqueueGenerate, enqueueVideo, enqueueImage, listLoraNames, jobsSnapshot, jobView, jobFiles, cancelJob, storePicture, readFile}

function keyHash(key) {
  return createHash('sha256').update(key).digest('hex')
}

/** Bearer check: sha256 of the presented key vs the configured hash, timing-safe. Fails closed when
 *  no hash is configured. */
export function keyMatches(header, storedHash) {
  const stored = String(storedHash || '').trim().toLowerCase()
  const presented = /^Bearer\s+(\S+)\s*$/i.exec(String(header || ''))?.[1]
  if (!stored || !presented) return false
  const a = Buffer.from(keyHash(presented))
  const b = Buffer.from(stored.length === 64 ? stored : keyHash(stored))
  return a.length === b.length && timingSafeEqual(a, b)
}

function workName(body) {
  const name = typeof body.work_name === 'string' ? body.work_name.trim().slice(0, 80) : ''
  return name ? {work_name:name} : {}
}

/** Data-URI images -> upload tokens via the same storePicture the chat tools use (max 8). */
async function decodeImages(list, deps) {
  if (list === undefined || list === null) return []
  if (!Array.isArray(list)) throw Error('images 必须是 data URI 数组')
  if (list.length > 8) throw Error('images 最多 8 张')
  const tokens = []
  for (const [i, item] of list.entries()) {
    const m = /^data:image\/(png|jpe?g|webp);base64,([\s\S]+)$/i.exec(String(item || ''))
    if (!m) throw Error('images[' + i + '] 只接受 data:image/(png|jpeg|webp);base64,... 形式')
    const data = Buffer.from(m[2].replace(/\s+/g, ''), 'base64')
    if (!data.length) throw Error('images[' + i + '] 没有图片数据')
    tokens.push(await deps.storePicture(data, '.' + (m[1].toLowerCase() === 'jpeg' ? 'jpg' : m[1].toLowerCase())))
  }
  return tokens
}

/** request.kind/mode -> the contract's kind. */
function jobKind(request) {
  if (request.kind === 'h3') return 'video'
  if (request.kind === 'ph3') return request.mode === 'edit' ? 'edit' : 'image'
  return 'image' // 'generate' (anima) and anything else
}

const iso = v => typeof v === 'number' && Number.isFinite(v) ? new Date(v > 1e12 ? v : v * 1000).toISOString() : undefined

function viewOut(view, deps) {
  const out = {id:view.id, kind:jobKind(view.request), state:view.state, stage:view.stage,
               created:iso(view.request.created), updated:iso(view.updated) ?? iso(view.request.created)}
  if (view.request.work_name) out.work_name = view.request.work_name
  if (view.position) out.position = view.position
  if (view.message) out.message = view.message
  if (view.state === 'done') {
    const files = deps.jobFiles(view)
    out.artifacts = files.map((path, index) => ({index, name:basename(path),
      content_type:TYPES[extname(path).toLowerCase()] || 'application/octet-stream',
      url:PREFIX + '/jobs/' + view.id + '/artifacts/' + index}))
  }
  return out
}

async function createJob(body, deps) {
  const kind = String(body.kind || '')
  if (!KINDS.includes(kind)) throw Error('kind 必须是 image / edit / video')
  const model = String(body.model ?? body.target ?? '').toLowerCase()
  const images = await decodeImages(body.images, deps)
  if (kind === 'video') {
    if (model && model !== 'h3') throw Error('kind "video" 只支持 model "h3"')
    const [width, height] = size(body.size, [1280, 704])
    const {id} = await enqueueWithLoras('h3', deps, () => deps.enqueueVideo({prompt:body.prompt, width, height,
      length:body.length ?? 124, seed:body.seed, mode:'t2v', loras:body.loras ?? [], ...workName(body)}))
    return id
  }
  const ph3 = kind === 'edit' || ['ph3', 'p-h3'].includes(model)
  if (kind === 'edit' && !images.length) throw Error('kind "edit" 需要 images（要编辑的源图，data URI）')
  if (!ph3) {
    if (model && model !== 'anima') throw Error('kind "image" 的 model 是 "anima" 或 "p-h3"')
    if (images.length) throw Error('anima 不接受图片输入；编辑/参照请用 kind "edit" 或 model "p-h3"')
    const [width, height] = size(body.size, [832, 1216])
    const {id} = await enqueueWithLoras('anima', deps, () => deps.enqueueGenerate({prompt:body.prompt, negative:body.negative ?? '',
      width, height, steps:body.steps ?? 34, cfg:body.cfg ?? 4.5, count:body.count ?? 1, seed:body.seed,
      loras:body.loras ?? [], fp8:true, ...workName(body)}))
    return id
  }
  const mode = kind === 'edit' ? 'edit' : images.length ? (body.mode === 'reference' ? 'reference' : 'edit') : 't2i'
  const [width, height] = size(body.size, mode === 'edit' ? [0, 0] : [2912, 2912])
  const {id} = await enqueueWithLoras('ph3', deps, () => deps.enqueueImage({prompt:body.prompt, mode, images, width, height,
    count:body.count ?? 1, seed:body.seed, loras:body.loras ?? [], ...workName(body)}))
  return id
}

/** Builds the raw (req, res) webServer handler; deps default to the real job queue functions. */
export function createAutomationHandler(overrides = {}) {
  const deps = {...real, ...overrides}
  return async (req, res) => {
    const send = (data, status = 200, headers = {}) => {
      res.writeHead(status, {'content-type':'application/json', 'cache-control':'no-store', ...headers})
      res.end(JSON.stringify(data))
    }
    try {
      if (!keyMatches(req.headers.authorization, overrides.keyHash ?? process.env.AUTOMATION_API_KEY_HASH))
        return send({ok:false, error:'invalid or missing API key'}, 401)
      const url = new URL(req.url, 'http://local')
      const path = url.pathname.slice(PREFIX.length) || '/'
      const parts = path.split('/').filter(Boolean)
      if (req.method === 'GET' && path === '/') return send({ok:true, service:'daraskDSH automation', version:1})
      if (req.method === 'GET' && path === '/status') {
        const {jobs} = await deps.jobsSnapshot()
        return send({ok:true, paused:false, queued:jobs.filter(j => j.state === 'queued').length,
                     running:jobs.filter(j => j.state === 'running').length, kinds:KINDS})
      }
      if (req.method === 'GET' && path === '/loras') {
        const model = String(url.searchParams.get('model') || 'anima').toLowerCase()
        if (!['anima', 'ph3', 'p-h3', 'h3'].includes(model)) return send({ok:false, error:'model 必须是 anima / ph3 / h3'}, 400)
        return send({loras:await deps.listLoraNames(model === 'p-h3' ? 'ph3' : model)})
      }
      if (parts[0] === 'jobs' && parts.length === 1) {
        if (req.method === 'GET') {
          const {jobs} = await deps.jobsSnapshot()
          return send({ok:true, jobs:jobs.map(view => viewOut(view, deps))})
        }
        if (req.method === 'POST') {
          if (req.headers['content-type']?.split(';')[0] !== 'application/json')
            return send({ok:false, error:'需要 application/json 请求体'}, 415)
          let bodySize = 0
          const chunks = []
          for await (const chunk of req) {
            bodySize += chunk.length
            if (bodySize > MAX_BODY) return send({ok:false, error:'请求体过大'}, 413)
            chunks.push(chunk)
          }
          const body = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
          const id = await createJob(body, deps)
          const view = await deps.jobView(id)
          return send({ok:true, job:view ? viewOut(view, deps) : {id}}, 201)
        }
      }
      if (parts[0] === 'jobs' && parts.length >= 2) {
        const id = parts[1]
        if (parts.length === 2 && req.method === 'GET') {
          const view = await deps.jobView(id)
          return view ? send({ok:true, job:viewOut(view, deps)}) : send({ok:false, error:'job not found: ' + id}, 404)
        }
        if (parts.length === 2 && req.method === 'DELETE') {
          await deps.cancelJob(id)
          return send({ok:true})
        }
        if (parts[2] === 'cancel' && parts.length === 3 && req.method === 'POST') {
          await deps.cancelJob(id)
          return send({ok:true})
        }
        if (parts[2] === 'artifacts' && parts.length === 4 && req.method === 'GET') {
          const view = await deps.jobView(id)
          const file = view && deps.jobFiles(view)[Number(parts[3])]
          const data = file && await deps.readFile(file).catch(() => null)
          if (!data) return send({ok:false, error:'artifact not found'}, 404)
          res.writeHead(200, {'content-type':TYPES[extname(file).toLowerCase()] || 'application/octet-stream',
            'content-length':data.length, 'cache-control':'private, max-age=86400',
            'content-disposition':'inline; filename="' + basename(file) + '"'})
          return res.end(data)
        }
      }
      return send({ok:false, error:'not found'}, 404)
    } catch (error) {
      send({ok:false, error:error.message}, 400)
    }
  }
}

/** Registers the /automation/v1 prefix route on the Harness web server (same process, same queue). */
export function registerAutomationApi(web) {
  web.effect(() => web.webServer.register({kind:'prefix', path:PREFIX, handler:createAutomationHandler()}), 'Automation API')
}
