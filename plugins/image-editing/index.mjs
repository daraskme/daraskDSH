import {mediaToolServices} from '../media-shared/tool-support.mjs'
import {registerImageEditing} from './tool.mjs'

export const name = 'dsh-image-editing'

export function apply(ctx) {
  ctx.inject(mediaToolServices, registerImageEditing)
}
