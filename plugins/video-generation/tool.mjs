import {defineTool} from '@deepseek-ai/dsh-tools'
import {enqueueVideo} from '../media-shared/jobs.mjs'
import {size, workspace, picture, finish, output, LORAS, withLoras, TIMEOUT_MS} from '../media-shared/tool-support.mjs'

export function registerVideoGeneration(ctx) {
  ctx.tools.register(defineTool({
    name:'video_generate',
    description:'Generate a video with sound (MiniMax H3, MP4 at 24 fps) on the local GPU and save it into the chat workspace ' +
      '(generated/<job>/). Describe both picture and sound in English. Without pictures it is text to video; first_frame and/or ' +
      'last_frame (workspace image paths) anchor the start/end frame (image to video); references (1-9 workspace image paths) ' +
      'are subjects the prompt names as <Picture 1>, <Picture 2>... Default 1280x704, 124 frames (~5 s, about 1.5 min). Up to 1440 frames (60 s): over 362 frames it is made in ~15 s segments that continue each other (about 2.5 min per segment).',
    parameters:{
      prompt:{type:'string', required:true},
      first_frame:{type:'string'}, last_frame:{type:'string'},
      references:{type:'array', items:{type:'string'}},
      size:{type:'string', description:'WIDTHxHEIGHT, multiples of 32, at most 1344x768 in area (e.g. 1280x704, 704x1280, 960x960)'},
      length:{type:'integer', description:'frames at 24 fps (5-1440), snapped up to 17k+5; most efficient long lengths: 702, 1042, 1382'},
      seed:{type:'integer'},
      loras:LORAS,
    },
    output, timeoutMs:TIMEOUT_MS + 60000, isConcurrencySafe:() => true,
    execute:async (args, exec) => {
      const root = await workspace(ctx, exec)
      const [width, height] = size(args.size, [1280, 704])
      const body = {prompt:args.prompt, width, height, length:args.length ?? 124, seed:args.seed, mode:'t2v', loras:args.loras ?? []}
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
      const {id} = await withLoras('h3', () => enqueueVideo(body))
      return JSON.stringify(await finish(ctx, exec, id, root))
    },
  }))
}
