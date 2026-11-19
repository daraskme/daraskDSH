import {registerMangaApi} from './api.mjs'

export const name = 'dsh-manga-studio'

export function apply(ctx) {
  ctx.inject(['connection'],registerMangaApi)
}
