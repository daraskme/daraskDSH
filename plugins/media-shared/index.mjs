import {registerJobApi} from './jobs.mjs'

export const name = 'dsh-media-shared'

export function apply(ctx) {
  ctx.inject(['connection'], registerJobApi)
}
