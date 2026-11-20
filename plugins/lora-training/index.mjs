import {animaToolServices, registerAnimaTool} from './anima-tool.mjs'

export const name = 'dsh-anima-lora-trainer'

export function apply(ctx) {
  ctx.inject(animaToolServices, registerAnimaTool)
}
