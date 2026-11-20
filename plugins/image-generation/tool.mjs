import {defineTool} from '@deepseek-ai/dsh-tools'
import {enqueueGenerate, enqueueImage} from '../media-shared/jobs.mjs'
import {size, workspace, picture, finish, output, LORAS, withLoras, TIMEOUT_MS} from '../media-shared/tool-support.mjs'

export function registerImageGeneration(ctx) {
  ctx.tools.register(defineTool({
    name:'image_generate',
    description:'Generate images on the local GPU and save them into the chat workspace (generated/<job>/). ' +
      'model "anima": anime-style text to image (Danbooru-style tags work well; negative prompt, steps, cfg and Anima LoRAs by name). ' +
      'model "p-h3": photoreal/general text to image up to 2912x2912 (natural-language English prompts), and with images: ' +
      'mode "edit" edits images[0] following the prompt (other images are references, called <Picture 2>, <Picture 3>...; size defaults to the edited image), ' +
      'mode "reference" makes a new picture of the subjects in images (refer to them as <Picture 1>, <Picture 2>... in the prompt). ' +
      'For a dedicated editing workflow use image_edit. Image paths are relative to the workspace. Several calls queue on one GPU; each returns when its own job is done.',
    parameters:{
      model:{type:'string', enum:['anima', 'p-h3'], required:true},
      prompt:{type:'string', required:true},
      mode:{type:'string', enum:['text', 'edit', 'reference']},
      images:{type:'array', items:{type:'string'}},
      size:{type:'string', description:'WIDTHxHEIGHT; Anima default 832x1216 (multiples of 16, up to 2048), P-H3 default 2912x2912 (multiples of 32)'},
      count:{type:'integer'}, seed:{type:'integer'},
      negative:{type:'string'}, steps:{type:'integer'}, cfg:{type:'number'},
      loras:LORAS,
    },
    output, timeoutMs:TIMEOUT_MS + 60000, isConcurrencySafe:() => true,
    execute:async (args, exec) => {
      const root = await workspace(ctx, exec)
      let id
      if (args.model === 'anima') {
        if (args.images?.length) throw Error('Anima は画像入力に対応していません。編集・参照は model "p-h3" を使ってください')
        const [width, height] = size(args.size, [832, 1216])
        ;({id} = await withLoras('anima', () => enqueueGenerate({prompt:args.prompt, negative:args.negative ?? '', width, height,
          steps:args.steps ?? 34, cfg:args.cfg ?? 4.5, count:args.count ?? 1, seed:args.seed, loras:args.loras ?? [], fp8:true})))
      } else {
        const images = []
        for (const path of args.images ?? []) images.push(await picture(root, path))
        const mode = images.length ? (args.mode === 'reference' ? 'reference' : 'edit') : 't2i'
        const [width, height] = size(args.size, mode === 'edit' ? [0, 0] : [2912, 2912])
        ;({id} = await withLoras('ph3', () => enqueueImage({prompt:args.prompt, mode, images, width, height, count:args.count ?? 1,
          seed:args.seed, loras:args.loras ?? []})))
      }
      return JSON.stringify(await finish(ctx, exec, id, root))
    },
  }))
}
