import {animaToolServices,registerAnimaTool} from './anima-tool.mjs'
import {registerJobApi} from './jobs.mjs'

export const name = 'dsh-anima-lora-trainer'

export function apply(ctx) {
  ctx.inject(animaToolServices,registerAnimaTool)
  ctx.inject(['connection'],registerJobApi)
}
