import {copyFile, mkdir, readFile, realpath} from 'node:fs/promises'
import {basename, extname, join, relative, resolve, sep} from 'node:path'
import {cancelJob, jobFiles, jobView, listLoraNames, storePicture} from './jobs.mjs'

export const mediaToolServices = ['tools', 'workingDirectory']
const TIMEOUT_MS = 60 * 60 * 1000
const POLL_MS = 2000

const size = (value, fallback) => {
  if (value === undefined || value === null || value === '') return fallback
  const match = /^(\d+)\s*[x×]\s*(\d+)$/.exec(String(value))
  if (!match) throw Error('size は「幅x高さ」で指定してください（例: 832x1216）')
  return [Number(match[1]), Number(match[2])]
}

async function workspace(ctx, exec) {
  return exec.agent === undefined ? process.cwd() : ctx.workingDirectory.ensure(exec.agent, exec.signal)
}

/** A picture inside the chat workspace -> an upload token (the job never reads outside the workspace). */
async function picture(root, path) {
  if (typeof path !== 'string' || !path.trim()) throw Error('画像のパスを指定してください')
  const real = await realpath(resolve(root, path.trim())).catch(() => null)
  const base = await realpath(root)
  if (!real || (real !== base && !real.startsWith(base + sep))) throw Error('ワークスペース内の画像を指定してください: ' + path)
  return storePicture(await readFile(real), extname(real))
}

async function finish(ctx, exec, id, root) {
  const deadline = Date.now() + TIMEOUT_MS
  try {
    for (;;) {
      exec.signal.throwIfAborted()
      const view = await jobView(id)
      if (!view) throw Error('ジョブが見つかりません: ' + id)
      if (view.state === 'done') {
        const target = join(root, 'generated', id)
        await mkdir(target, {recursive:true})
        const files = []
        for (const file of jobFiles(view)) {
          const copy = join(target, basename(file))
          await copyFile(file, copy)
          files.push(relative(root, copy).split(sep).join('/'))
        }
        return {job:id, files, seeds:(view.images || []).map(item => item.seed), load_seconds:view.load_seconds,
                total_seconds:view.total_seconds ?? Math.round((view.updated - view.started) || 0)}
      }
      if (['error', 'cancelled'].includes(view.state)) throw Error((view.state === 'error' ? '生成に失敗しました: ' : '中止されました: ') + (view.message || id))
      if (Date.now() > deadline) throw Error('生成が時間内に終わりませんでした: ' + id)
      await new Promise((done, fail) => {
        const timer = setTimeout(done, POLL_MS)
        exec.signal.addEventListener('abort', () => { clearTimeout(timer); fail(exec.signal.reason) }, {once:true})
      })
    }
  } catch (error) {
    if (exec.signal.aborted) await cancelJob(id).catch(() => {})
    throw error
  }
}

const output = {schema:{type:'string'}, render:(_, value) => [{type:'text', text:value}]}
const LORAS = {type:'array', description:'[{name, strength}] LoRAs of that model, names relative to models/loras (an unknown name lists the usable ones)',
  items:{type:'object', properties:{name:{type:'string', required:true}, strength:{type:'number'}}, additionalProperties:false}}

/** enqueue(), adding the usable LoRA names to a LoRA error. */
async function withLoras(model, enqueue) {
  try {
    return await enqueue()
  } catch (error) {
    if (!/LoRA/.test(error.message)) throw error
    throw Error(error.message + '（使える LoRA: ' + ((await listLoraNames(model)).join(', ') || 'なし') + '）')
  }
}

export {size, workspace, picture, finish, output, LORAS, withLoras, TIMEOUT_MS}
