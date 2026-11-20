import {mediaToolServices} from '../media-shared/tool-support.mjs'
import {registerImageGeneration} from './tool.mjs'

export const name = 'dsh-image-generation'

export function apply(ctx) {
  ctx.inject(mediaToolServices, registerImageGeneration)
}
