import {registerJobApi} from './jobs.mjs'
import {registerAutomationApi} from './automation.mjs'

export const name = 'dsh-media-shared'

export function apply(ctx) {
  ctx.inject(['connection'], registerJobApi)
  ctx.inject(['webServer'], registerAutomationApi)
}
