import {defineTool} from '@deepseek-ai/dsh-tools'
import {enqueueImage} from '../media-shared/jobs.mjs'
import {size, workspace, picture, finish, output, LORAS, withLoras, TIMEOUT_MS} from '../media-shared/tool-support.mjs'

export function registerImageEditing(ctx) {
  ctx.tools.register(defineTool({
    name:'image_edit',
    description:'Edit an existing picture with P-H3 on the local GPU. Describe what to change and which details to preserve. ' +
      'Optional references are <Picture 2>, <Picture 3>, and so on in the prompt. Paths are relative to the chat workspace.',
    parameters:{
      image:{type:'string',required:true}, prompt:{type:'string',required:true},
      references:{type:'array',items:{type:'string'}},
      size:{type:'string',description:'WIDTHxHEIGHT; defaults to the input image size'},
      count:{type:'integer'}, seed:{type:'integer'}, loras:LORAS,
    },
    output, timeoutMs:TIMEOUT_MS + 60000, isConcurrencySafe:() => true,
    execute:async (args, exec) => {
      const root = await workspace(ctx, exec)
      const images = [await picture(root, args.image)]
      for (const path of args.references ?? []) images.push(await picture(root, path))
      const [width, height] = size(args.size, [0, 0])
      const {id} = await withLoras('ph3', () => enqueueImage({
        mode:'edit', images, prompt:args.prompt, width, height, count:args.count ?? 1,
        seed:args.seed, loras:args.loras ?? [],
      }))
      return JSON.stringify(await finish(ctx, exec, id, root))
    },
  }))
}
