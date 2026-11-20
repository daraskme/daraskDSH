import {execFile} from 'node:child_process'
import {fileURLToPath} from 'node:url'
import {defineTool} from '@deepseek-ai/dsh-tools'

const python = fileURLToPath(new URL('../.venv/bin/python', import.meta.url))
const script = fileURLToPath(new URL('./search.py', import.meta.url))
export const inject = ['web', 'tools']

export function search(query, maxResults, xOnly, signal) {
  if (!query?.trim()) throw new Error('検索語が必要です')
  const limit = Math.min(Math.max(1, maxResults || 8), 30)
  return new Promise((resolve, reject) => {
    const proc = execFile(python, [script], {signal, timeout: 45000, maxBuffer: 2 ** 20}, (error, stdout, stderr) => {
      if (error) return reject(new Error(stderr.trim().split('\n').at(-1) || 'DDGS search failed', {cause: error}))
      try { resolve(JSON.parse(stdout)) } catch (e) { reject(e) }
    })
    proc.stdin.on('error', error => { if (error.code !== 'EPIPE') reject(error) })
    proc.stdin.end(JSON.stringify({query, limit, xOnly}))
  })
}

export function apply(ctx) {
  ctx.web.registerSearchProvider({id: 'ddgs', available: () => true,
    search: ({query, maxResults}, signal) => search(query, maxResults, false, signal)})
  ctx.tools.register(defineTool({
    name: 'x_search', description: 'Search public X / Twitter posts via Yahoo Japan Realtime Search and the DDGS fallback. Returns post links and excerpts. Index coverage and freshness vary.',
    parameters: {query: {type: 'string', required: true, description: 'Search terms for public posts.'}},
    output: {schema: {type: 'string'}, render: (_, value) => [{type: 'text', text: value}]},
    timeoutMs: 50000, isConcurrencySafe: () => true,
    execute: async ({query}, exec) => JSON.stringify((await search(query, 8, true, exec.signal)).sources)
  }))
}
