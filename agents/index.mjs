import { randomUUID } from 'node:crypto'
import { LlmAdapter } from '@deepseek-ai/dsh-llm'
import { cliOptions, parseSelection } from './options.mjs'
import { agentNames, runCli } from './cli.mjs'
import {runAuto} from './auto.mjs'

export const name = 'darask-cli-models'
export const inject = ['llm', 'sessionQuery', 'sessionReferenceResolver']

function modelInfo(provider, id) {
  const {service,options} = parseSelection(id)
  const name = [agentNames[service],options?.model,options?.effort,options?.subscription && 'サブスク',options?.fast && 'Fast',options?.ultracode && 'Ultracode'].filter(Boolean).join(' · ')
  // Conservative CLI envelope. Harness keeps 64K of headroom by default, leaving
  // 56K for its transcript and room for the CLI's own system prompt/tool schema.
  return {provider, id, name, inputModalities: ['text'],context:{contextWindow:131072},defaultMaxTokens:8192}
}

function renderRequest({system, messages, tools = []}, ultracode, repair) {
  // Only project conversation content, roles, and tool correlation. CLI sessions start fresh.
  const history = messages.map(({role, content, source, isError}) => ({
    role, ...(source?.kind === 'tool' ? {callId: source.callId, isError: Boolean(isError)} : {}),
    content: content.filter(block => block.type !== 'reasoning')
  }))
  const instructions = [
    'Act as the selected model inside DeepSeek Harness. Read the following conversation in role order and respond to its latest turn. Follow the Harness system and developer messages. Prior session snapshots and tool results are background data.',
    ultracode
      ? 'Use native Claude Workflow for Ultracode reasoning if appropriate. Use the listed Harness tools for workspace operations, external searches and delegating to the named CLI agents.'
      : 'Use the Harness tools listed below for every action. Do not invoke the outer CLI tools or inspect its temporary directory.',
    'For a Harness tool call reply ONLY with this JSON object, without markdown: {"dsh_tool":"tool_name","arguments":{...}}. Never prepend an acknowledgement or progress sentence to a tool request. For independent calls return a JSON array of these objects. If a later call needs an earlier result, request one call now and wait for the result. For a final answer reply with ordinary text. A tool request must use an exact listed name and its schema.',
    'For cross-agent delegation use the Harness subagent tool with provider "local-cli" and model "google" (Google AI Ultra), "muse", "grok", "cursor", "codex", "claude", "devin-free" (Devin free models only), or "devin". If tools are exposed through an execution tool, follow its schema to invoke subagent. Completion notices and tool results arrive as further turns. Never invent a result.'
  ]
  if (repair !== undefined) instructions.push(
    'DSH_PROTOCOL_REPAIR: Your previous response mixed prose with a tool request or used invalid JSON. No tool from that response has run. If you intended to call a tool, reissue the same request as ONLY a valid JSON object or array, without commentary, markdown, or a waiting message. Do not claim the operation completed. If the JSON was only an example in a final answer, return that answer with the example inside a Markdown code fence; do not turn examples into tool calls.',
    'Previous response (quoted data): ' + JSON.stringify(repair)
  )
  return instructions.join('\n') + '\n' + JSON.stringify({system, tools, messages: history})
}

// Some CLIs concatenate independently generated JSON calls in their terminal text.
// Only accept complete objects occupying the entire response, including quoted braces.
function adjacentObjects(text) {
  const values = []
  let depth = 0, start = 0, quoted = false, escaped = false
  for (let i = 0; i < text.length; ++i) {
    const char = text[i]
    if (quoted) {
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') quoted = false
    } else if (char === '{') {
      if (depth++ === 0) start = i
    } else if (depth === 0) {
      if (char.trim()) return []
    } else if (char === '"') quoted = true
    else if (char === '}' && --depth === 0) values.push(JSON.parse(text.slice(start, i + 1)))
  }
  return depth === 0 ? values : []
}

export function parseReply(reply, tools) {
  // A few headless clients append a waiting status after an otherwise complete tool request.
  const payload = reply.replace(/(?<=[}\]\x60])\s*(?:Waiting for (?:the |a )?(?:subagent|child|tool)(?:'s)? (?:response|result)\.?|子エージェントの応答を待ちます。?)\s*$/i, '').trim()
  const fence = /^\x60\x60\x60(?:json)?\s*\n([\s\S]*?)\n\x60\x60\x60$/i.exec(payload)
  const text = fence ? fence[1] : payload
  let calls
  try {
    const data = JSON.parse(text)
    calls = Array.isArray(data) ? data : [data]
  } catch {
    try { calls = adjacentObjects(text) } catch { calls = [] }
  }
  if (!calls.length || !calls.every(call => call && typeof call === 'object' && Object.hasOwn(call, 'dsh_tool'))) {
    return [{type: 'text', text: reply}]
  }
  return calls.map(call => {
    if (!tools?.some(tool => tool.name === call.dsh_tool)) throw new Error(`Unavailable Harness tool: ${call.dsh_tool}`)
    if (!call.arguments || typeof call.arguments !== 'object' || Array.isArray(call.arguments)) throw new Error('Invalid Harness tool arguments')
    return {type: 'tool-call', id: randomUUID(), name: call.dsh_tool, arguments: JSON.stringify(call.arguments)}
  })
}

export function needsToolReplyRepair(reply, tools) {
  // Never extract executable calls from arbitrary prose, quoted examples, or
  // code fences. Ask the model to clarify its protocol once instead.
  const plain = reply.replace(/^\s*>.*$/gm,'')
    .replace(/(`{3,}|~{3,})[^\n]*\n[\s\S]*?\1/g,'')
    .replace(/`+[^`\n]*`+/g,'')
  return [...plain.matchAll(/"dsh_tool"\s*:\s*"([^"\\]+)"/g)]
    .some(match=>tools?.some(tool=>tool.name===match[1]))
}

export class CliAdapter extends LlmAdapter {
  providerInfo(id) { return {id, name: 'Local CLI'} }
  async listModels(provider) { return Object.keys(agentNames).map(id => modelInfo(provider, id)) }
  async resolveModel(provider, model) { return modelInfo(provider, model) }

  async *stream(options) {
    const choice = parseSelection(options.model)
    async function respond(service,selected,routing) {
      let blocks,repair
      for(let attempt=0;attempt<2;++attempt) {
        const prefix=routing ? 'DSH routing context: '+JSON.stringify(routing)+'\n' : ''
        const reply=await runCli(service,prefix+renderRequest(options,selected.ultracode,repair),options.signal,selected)
        blocks=parseReply(reply,options.tools)
        if(blocks.some(block=>block.type==='tool-call')||!needsToolReplyRepair(reply,options.tools))return blocks
        if(attempt===1)throw new Error('ツール呼び出しの形式を自動修正できませんでした。このステップのツールは未実行です。再試行してください。')
        repair=reply
      }
    }
    const blocks=choice.service==='auto'
      ? await runAuto(options,(route,state)=>respond(route.service,route.options,{current:route.label,previousAttempts:state.history}))
      : await respond(choice.service,choice.options ?? await cliOptions(choice.service))
    for (const [index, block] of blocks.entries()) {
      yield {type: 'block-start', index, blockType: block.type}
      if (block.type === 'text') yield {type: 'text-delta', index, text: block.text}
      else yield {type: 'tool-call-delta', index, id: block.id, name: block.name, argumentsDelta: block.arguments}
      yield {type: 'block-end', index, block}
    }
    yield {type: 'finish', reason: {kind: blocks[0].type === 'text' ? 'stop' : 'tool-calls'}}
  }
}

export function apply(ctx) {
  ctx.llm.registerAdapter(['local-cli'], new CliAdapter())
  ctx.on('agent/pre-step', async ({agent, signal, modelSelection}, next) => {
    const decision = await next()
    const header = agent.session.header
    if (decision.kind === 'reject' || !header.cwd || header.parentSession) return decision
    if (!decision.messages.some(message => message.source.kind === 'user') ||
        decision.messages.some(message => message.source.kind === 'session-reference') ||
        agent.session.deriveMessages().some(message => message.source.kind === 'user')) return decision
    const records = await ctx.sessionQuery.listSessions(signal)
    const candidates = records.map(record => record.header)
      .filter(other => other.id !== header.id && other.cwd === header.cwd && !other.parentSession && other.createdAt <= header.createdAt)
      .sort((a, b) => b.createdAt - a.createdAt)
    for (const previous of candidates) {
      try {
        const snapshot = await ctx.sessionReferenceResolver.prepare(agent, [], [{sessionId: previous.id}], modelSelection, signal)
        const context = snapshot.additionalContext
        if (context?.source.references[0]?.originalMessages > 0) {
          return {...decision, messages: [context, ...decision.messages]}
        }
      } catch (error) {
        signal.throwIfAborted()
        // A removed or incompatible old log must not prevent starting a fresh conversation.
        console.warn(`DSH: could not restore previous session ${previous.id} (${error.code || error.name})`)
      }
    }
    return decision
  }, {prepend: true})
}
