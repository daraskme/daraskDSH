// Chat tools for every agent (any registered LLM): image_generate (Anima or P-H3: text to image, edit, reference
// generation) and video_generate (MiniMax H3: text, first/last frame, reference pictures). They put the job on the
// same GPU queue as the sidebar panels, wait for that job only, and copy the results into the chat workspace.
import {copyFile, mkdir, readFile, realpath} from 'node:fs/promises'
import {basename, extname, isAbsolute, join, relative, resolve, sep} from 'node:path'
import {defineTool} from '@deepseek-ai/dsh-tools'
import {cancelJob, enqueueGenerate, enqueueImage, enqueueVideo, jobFiles, jobView, listLoraNames, storePicture} from './jobs.mjs'

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

export function registerMediaTools(ctx) {
  ctx.tools.register(defineTool({
    name:'image_generate',
    description:'Use this (not ComfyUI) for any image generation or image editing request. Generate images on the local GPU and save them into the chat workspace (generated/<job>/). ' +
      'model "anima": anime-style text to image (Danbooru-style tags work well; negative prompt, steps, cfg and Anima LoRAs by name). ' +
      'model "p-h3": photoreal/general text to image up to 2912x2912 (natural-language English prompts), and with images: ' +
      'mode "edit" edits images[0] following the prompt (other images are references, called <Picture 2>, <Picture 3>...; size defaults to the edited image), ' +
      'mode "reference" makes a new picture of the subjects in images (refer to them as <Picture 1>, <Picture 2>... in the prompt). ' +
      'Image paths are relative to the workspace. Several calls queue on one GPU; each returns when its own job is done.',
    parameters:{
      model:{type:'string', enum:['anima', 'p-h3'], required:true},
      prompt:{type:'string', required:true},
      mode:{type:'string', enum:['text', 'edit', 'reference']},
      images:{type:'array', items:{type:'string'}},
      size:{type:'string', description:'WIDTHxHEIGHT; Anima default 832x1216 (multiples of 16, up to 2048), P-H3 default 2912x2912 (multiples of 32)'},
      count:{type:'integer'}, seed:{type:'integer'},
      negative:{type:'string'}, steps:{type:'integer'}, cfg:{type:'number'},
      loras:{type:'array', description:'Anima only: [{name, strength}] with names relative to models/loras',
        items:{type:'object', properties:{name:{type:'string', required:true}, strength:{type:'number'}}, additionalProperties:false}},
    },
    output, timeoutMs:TIMEOUT_MS + 60000, isConcurrencySafe:() => true,
    execute:async (args, exec) => {
      const root = await workspace(ctx, exec)
      let id
      if (args.model === 'anima') {
        if (args.images?.length) throw Error('Anima は画像入力に対応していません。編集・参照は model "p-h3" を使ってください')
        const [width, height] = size(args.size, [832, 1216])
        try {
          ({id} = await enqueueGenerate({prompt:args.prompt, negative:args.negative ?? '', width, height, steps:args.steps ?? 34,
            cfg:args.cfg ?? 4.5, count:args.count ?? 1, seed:args.seed, loras:args.loras ?? [], fp8:true}))
        } catch (error) {
          if (!/LoRA/.test(error.message)) throw error
          throw Error(error.message + '（使える LoRA: ' + ((await listLoraNames()).join(', ') || 'なし') + '）')
        }
      } else {
        const images = []
        for (const path of args.images ?? []) images.push(await picture(root, path))
        const mode = images.length ? (args.mode === 'reference' ? 'reference' : 'edit') : 't2i'
        const [width, height] = size(args.size, mode === 'edit' ? [0, 0] : [2912, 2912])
        ;({id} = await enqueueImage({prompt:args.prompt, mode, images, width, height, count:args.count ?? 1, seed:args.seed}))
      }
      return JSON.stringify(await finish(ctx, exec, id, root))
    },
  }))

  ctx.tools.register(defineTool({
    name:'video_generate',
    description:'Use this (not ComfyUI) for any video generation request, including image to video. Generate a video with sound (MiniMax H3, MP4 at 24 fps) on the local GPU and save it into the chat workspace ' +
      '(generated/<job>/). Describe both picture and sound in English. Without pictures it is text to video; first_frame and/or ' +
      'last_frame (workspace image paths) anchor the start/end frame (image to video); references (1-9 workspace image paths) ' +
      'are subjects the prompt names as <Picture 1>, <Picture 2>... Default 1280x704, 124 frames (~5 s); takes about a minute.',
    parameters:{
      prompt:{type:'string', required:true},
      first_frame:{type:'string'}, last_frame:{type:'string'},
      references:{type:'array', items:{type:'string'}},
      size:{type:'string', description:'WIDTHxHEIGHT, multiples of 32, at most 1344x768 in area (e.g. 1280x704, 704x1280, 960x960)'},
      length:{type:'integer', description:'frames at 24 fps (5-362), snapped up to 17k+5'},
      seed:{type:'integer'},
    },
    output, timeoutMs:TIMEOUT_MS + 60000, isConcurrencySafe:() => true,
    execute:async (args, exec) => {
      const root = await workspace(ctx, exec)
      const [width, height] = size(args.size, [1280, 704])
      const body = {prompt:args.prompt, width, height, length:args.length ?? 124, seed:args.seed, mode:'t2v'}
      if (args.references?.length) {
        if (args.first_frame || args.last_frame) throw Error('references と first_frame / last_frame は同時に使えません')
        body.mode = 'ref'
        body.references = []
        for (const path of args.references) body.references.push(await picture(root, path))
      } else if (args.first_frame || args.last_frame) {
        body.mode = 'i2v'
        if (args.first_frame) body.first = await picture(root, args.first_frame)
        if (args.last_frame) body.last = await picture(root, args.last_frame)
      }
      const {id} = await enqueueVideo(body)
      return JSON.stringify(await finish(ctx, exec, id, root))
    },
  }))
}
