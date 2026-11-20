// Recommended native Anima LoRA settings per purpose (1024px, AdamW, batch 1).
// Style follows the user's reference style preset (rank 8, LR 4e-5, cross-attention
// 1e-5, 6% warmup, cosine with 4 restarts, timestep scale 1.2, no weight decay),
// which trained well on 19 images in ~600 steps. Character and concept train every
// module at rank 16 with a single cosine and a few more steps for identity/concept.
export const PURPOSES = {
  character: {
    label:'キャラ', targetSteps:800, rank:16,
    args:{'learning-rate':4e-5, warmup:0.06, 'lr-cycles':1, 'timestep-scale':1, 'weight-decay':0},
  },
  style: {
    label:'スタイル', targetSteps:600, rank:8,
    args:{'learning-rate':4e-5, 'cross-attn-lr':1e-5, warmup:0.06, 'lr-cycles':4, 'timestep-scale':1.2, 'weight-decay':0},
  },
  concept: {
    label:'コンセプト', targetSteps:700, rank:16,
    args:{'learning-rate':4e-5, warmup:0.06, 'lr-cycles':1, 'timestep-scale':1, 'weight-decay':0},
  },
}

// Caption instructions per purpose (originally from ComfyUI-Folder-LoRA dataset.py).
export const CAPTION_INSTRUCTIONS = {
  character:'Focus on one recognizable subject. Describe variable pose, action, expression, clothing, framing, lighting and background. Let the activation phrase carry persistent identity. Do not invent names or backstory.',
  style:'Caption the depicted subjects, arrangement, actions, objects, scenery and lighting in concrete terms. Let the activation phrase carry the target rendering style. Avoid artist names, medium labels and style tags.',
  concept:'Describe the visible event, interacting objects, subject relationships, spatial context and what changes around the target concept. Ground actions and physical details in the evidence. Do not invent causes.',
}

export const RESOLUTION = 1024

// Models the trainer handles. H3 / P-H3 follow diffusion-pipe's MiniMax H3 defaults (rank 32, AdamW 1e-4, 10 warmup
// steps, CFG-augmented); their pictures train at 768 / 1024 px and H3 videos at 512 px (train_lora.py).
export const MODELS = {
  anima:{label:'Anima', videos:false},
  h3:{label:'MiniMax H3（動画・画像）', videos:true, targetSteps:300, rank:32, resolution:768, args:{'learning-rate':1e-4}},
  ph3:{label:'P-H3 画像', videos:false, targetSteps:400, rank:32, resolution:1024, args:{'learning-rate':1e-4}},
}

/** Settings for a dataset of `images` files (videos count as files too): epochs aim at the step budget. */
export function recommend(purpose, images, model = 'anima') {
  const preset = PURPOSES[purpose]
  if (!preset) throw Error('目的を選んでください')
  const kind = MODELS[model]
  if (!kind) throw Error('モデルを選んでください')
  if (model !== 'anima') {
    const epochs = Math.max(2, Math.min(60, Math.round(kind.targetSteps / images)))
    return {model, purpose, label:preset.label, epochs, steps:epochs * images, rank:kind.rank, resolution:kind.resolution,
            args:{rank:kind.rank, epochs, ...kind.args}}
  }
  const epochs = Math.max(4, Math.min(60, Math.round(preset.targetSteps / images)))
  const args = {resolution:RESOLUTION, rank:preset.rank, epochs, ...preset.args}
  return {model, purpose, label:preset.label, epochs, steps:epochs * images, rank:preset.rank, resolution:RESOLUTION, args}
}
