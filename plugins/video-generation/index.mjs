import {mediaToolServices} from '../media-shared/tool-support.mjs'
import {registerVideoGeneration} from './tool.mjs'

export const name = 'dsh-video-generation'

export function apply(ctx) {
  ctx.inject(mediaToolServices, registerVideoGeneration)
}
