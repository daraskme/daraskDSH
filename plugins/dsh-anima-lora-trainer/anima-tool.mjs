import {fileURLToPath} from 'node:url'
import {defineTool} from '@deepseek-ai/dsh-tools'

const dsh = fileURLToPath(new URL('../../dsh', import.meta.url))
const timeoutMs = 12 * 60 * 60 * 1000
const flags = {
  folder:'folder', output:'output', resolution:'resolution', rank:'rank', steps:'steps',
  epochs:'epochs', learningRate:'learning-rate', seed:'seed', cacheDir:'cache-dir',
  config:'config', frontend:'frontend', metrics:'metrics',
}
const quote = value => "'" + value.replaceAll("'", "'\\''") + "'"

export const animaToolServices = ['shell', 'shellEnv', 'workingDirectory']

export function registerAnimaTool(ctx) {
  ctx.tools.register(defineTool({
    name:'anima_lora_train',
    description:'Train a ComfyUI-compatible Anima LoRA on the local CUDA GPU from a folder of images with same-stem .txt captions. Use a new .safetensors output path. Paths are relative to the current chat workspace. Specify steps or epochs. The training process is cancelled if the tool call is stopped.',
    parameters:{
      folder:{type:'string',required:true}, output:{type:'string',required:true},
      resolution:{type:'integer'}, rank:{type:'integer'}, steps:{type:'integer'}, epochs:{type:'integer'},
      learningRate:{type:'number'}, seed:{type:'integer'}, recursive:{type:'boolean'},
      cacheDir:{type:'string'}, config:{type:'string'}, frontend:{type:'string'}, metrics:{type:'string'},
    },
    output:{schema:{type:'string'},render:(_,value)=>[{type:'text',text:value}]},
    timeoutMs,
    isConcurrencySafe:()=>false,
    execute:async (args, exec) => {
      const workdir = exec.agent === undefined ? undefined : await ctx.workingDirectory.ensure(exec.agent, exec.signal)
      const sandboxPolicy = ctx.get('sandboxPolicy')?.resolve(exec.agent === undefined ? {} : {session:exec.agent.session})
      if (ctx.shell.sandboxMode !== undefined && sandboxPolicy === undefined) {
        throw new Error('Anima training needs the shell sandbox policy for this session')
      }

      const argv = [dsh, 'anima', 'train', '--summary-only', '--log-every=25']
      for (const [argument, option] of Object.entries(flags)) {
        if (args[argument] !== undefined) argv.push(`--${option}=${args[argument]}`)
      }
      if (args.recursive) argv.push('--recursive')

      const deadline = AbortSignal.timeout(timeoutMs)
      const signal = AbortSignal.any([exec.signal, deadline])
      const spec = ctx.shell.resolve({
        command:`exec ${argv.map(quote).join(' ')}`,
        workdir, signal, sandboxPolicy, dshEnv:ctx.shellEnv.collect(exec), onExpiry:'none',
      })
      const started = performance.now()
      const result = await (await ctx.shell.execute(spec)).result()
      exec.signal.throwIfAborted()
      if (deadline.aborted || result.timedOut) throw new Error('Anima training timed out')
      if (result.sandbox?.denied) throw new Error('The session sandbox denied Anima training file access')
      if (result.exitCode !== 0) {
        const detail = result.stderr.text.trim().split('\n').slice(-4).join('\n').slice(-2000)
        throw new Error(`Anima training failed (${result.exitCode ?? result.signal ?? 'unknown status'})${detail ? `\n${detail}` : ''}`)
      }

      const summary = JSON.parse(result.stdout.text.trimEnd().split('\n').at(-1))
      const median = summary.median_step_ms_after_first ?? summary.median_step_ms
      return JSON.stringify({
        output:summary.output, gpu:summary.gpu, images:summary.images, steps:summary.steps,
        cache_hits:summary.cache_hits, preprocessing_seconds:summary.preprocessing_seconds,
        training_seconds:summary.training_seconds, wall_seconds:(performance.now() - started) / 1000,
        median_step_ms:median, images_per_second:median > 0 ? 1000 / median : null,
      })
    },
  }))
}
