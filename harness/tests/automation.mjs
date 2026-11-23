// Exercises the /automation/v1 handler end-to-end over real HTTP with a stubbed enqueue layer —
// no GPU, no Harness. Covers auth, validation, LoRA passthrough, data-URI decoding and model mapping.
import test, {after} from 'node:test'
import assert from 'node:assert/strict'
import {once} from 'node:events'
import {createHash} from 'node:crypto'
import http from 'node:http'
import {createAutomationHandler} from '../../plugins/media-shared/automation.mjs'

const KEY = 'dshak_testkey1234567890'
const KEY_HASH = createHash('sha256').update(KEY).digest('hex')
const AUTH = {authorization:'Bearer ' + KEY}

const calls = []
const views = {}
const stored = []
const deps = {
  keyHash:KEY_HASH,
  enqueueGenerate:async body => { calls.push(['anima', body]); if (!body.prompt) throw Error('prompt required'); views['id-anima'] = viewOf('id-anima','generate'); return {id:'id-anima'} },
  enqueueVideo:async body => { calls.push(['h3', body]); if (!body.prompt) throw Error('prompt required'); views['id-video'] = viewOf('id-video','h3'); return {id:'id-video'} },
  enqueueImage:async body => { calls.push(['ph3', body]); if (!body.prompt) throw Error('prompt required'); views['id-ph3'] = viewOf('id-ph3','ph3',body.mode); return {id:'id-ph3'} },
  listLoraNames:async model => ({anima:['anima-a.safetensors','anima-b.safetensors'], ph3:['ph3-x.safetensors'], h3:['h3-v.safetensors']}[model] ?? []),
  jobsSnapshot:async () => ({jobs:Object.values(views)}),
  jobView:async id => views[id] ?? null,
  jobFiles:view => (view.images || []).map(item => '/out/' + item.file),
  cancelJob:async id => { calls.push(['cancel', id]) },
  storePicture:async (data, ext) => { stored.push([data, ext]); return 'tok' + stored.length + ext },
  readFile:async path => path.includes('000-123.png') ? Buffer.from('png-bytes') : Buffer.from('mp4-bytes'),
}
function viewOf(id, kind, mode, extra = {}) {
  return {id, request:{kind, mode, created:1700000000000, ...extra.request}, state:'queued', stage:'queued', ...extra}
}

const server = http.createServer(createAutomationHandler(deps))
server.listen(0, '127.0.0.1')
await once(server, 'listening')
after(() => server.close())
const base = 'http://127.0.0.1:' + server.address().port + '/automation/v1'

async function api(path, {method = 'GET', body, headers = {}} = {}) {
  const res = await fetch(base + path, {method, headers:{...AUTH, ...(body !== undefined ? {'content-type':'application/json'} : {}), ...headers},
    body:body === undefined ? undefined : JSON.stringify(body)})
  const text = await res.text()
  let data
  try { data = JSON.parse(text) } catch { data = text }
  return [res.status, data, res]
}

test('missing or wrong key -> 401, correct key passes', async () => {
  const noKey = await fetch(base, {})
  assert.equal(noKey.status, 401)
  const badKey = await fetch(base, {headers:{authorization:'Bearer wrong'}})
  assert.equal(badKey.status, 401)
  const [status, data] = await api('/')
  assert.equal(status, 200)
  assert.equal(data.ok, true)
})

test('job body validation errors -> 400', async () => {
  assert.equal((await api('/jobs', {method:'POST', body:{kind:'image', model:'anima'}}))[0], 400) // missing prompt
  assert.equal((await api('/jobs', {method:'POST', body:{kind:'wat', prompt:'x'}}))[0], 400) // bad kind
  assert.equal((await api('/jobs', {method:'POST', body:{kind:'image', prompt:'x', size:'abc'}}))[0], 400) // bad size
  const [, badModel] = await api('/jobs', {method:'POST', body:{kind:'video', model:'anima', prompt:'x'}})
  assert.match(badModel.error, /h3/)
  const [nj] = await api('/jobs', {method:'POST', body:'not-json', headers:{'content-type':'text/plain'}})
  assert.equal(nj, 415)
})

test('loras passthrough to enqueue functions unchanged', async () => {
  calls.length = 0
  const loras = [{name:'anima-a.safetensors', strength:0.8}]
  const [status] = await api('/jobs', {method:'POST', body:{kind:'image', prompt:'x', loras, work_name:'批A'}})
  assert.equal(status, 201)
  assert.deepEqual(calls[0][1].loras, loras)
  assert.equal(calls[0][1].work_name, '批A')
  calls.length = 0
  await api('/jobs', {method:'POST', body:{kind:'video', prompt:'x', loras:[{name:'h3-v.safetensors'}]}})
  assert.deepEqual(calls[0][1].loras, [{name:'h3-v.safetensors'}])
})

test('LoRA errors return 400 listing usable names', async () => {
  const bad = {...deps, enqueueGenerate:async () => { throw Error('LoRA が見つかりません: nope') }}
  const s2 = http.createServer(createAutomationHandler(bad))
  s2.listen(0, '127.0.0.1')
  await once(s2, 'listening')
  const res = await fetch('http://127.0.0.1:' + s2.address().port + '/automation/v1/jobs',
    {method:'POST', headers:{...AUTH, 'content-type':'application/json'}, body:JSON.stringify({kind:'image', prompt:'x', loras:[{name:'nope'}]})})
  assert.equal(res.status, 400)
  const data = await res.json()
  assert.match(data.error, /anima-a\.safetensors/)
  s2.close()
})

test('edit jobs decode data URIs through storePicture; max 8 / missing source -> 400', async () => {
  calls.length = 0; stored.length = 0
  const uri = 'data:image/png;base64,' + Buffer.from('fakepng').toString('base64')
  const [status] = await api('/jobs', {method:'POST', body:{kind:'edit', prompt:'改色', images:[uri]}})
  assert.equal(status, 201)
  assert.equal(stored.length, 1)
  assert.deepEqual(stored[0][0], Buffer.from('fakepng'))
  assert.equal(stored[0][1], '.png')
  assert.deepEqual(calls[0][1].images, ['tok1.png'])
  assert.equal(calls[0][1].mode, 'edit')

  const [s1, e1] = await api('/jobs', {method:'POST', body:{kind:'edit', prompt:'x'}})
  assert.equal(s1, 400); assert.match(e1.error, /images/)
  const [s2, e2] = await api('/jobs', {method:'POST', body:{kind:'edit', prompt:'x', images:Array(9).fill(uri)}})
  assert.equal(s2, 400); assert.match(e2.error, /8/)
  const [s3, e3] = await api('/jobs', {method:'POST', body:{kind:'edit', prompt:'x', images:['data:image/svg+xml;base64,PHN2Zy8+']}})
  assert.equal(s3, 400); assert.match(e3.error, /data:image/)
})

test('loras endpoint maps model names', async () => {
  const [, a] = await api('/loras?model=anima')
  assert.deepEqual(a.loras, ['anima-a.safetensors','anima-b.safetensors'])
  const [, p] = await api('/loras?model=ph3')
  assert.deepEqual(p.loras, ['ph3-x.safetensors'])
  const [, h] = await api('/loras?model=h3')
  assert.deepEqual(h.loras, ['h3-v.safetensors'])
  assert.equal((await api('/loras?model=bad'))[0], 400)
})

test('job list / detail / artifact download / cancel', async () => {
  views['j-done'] = {id:'j-done', request:{kind:'ph3', mode:'edit', created:1700000000000, work_name:'作品'},
    state:'done', stage:'done', updated:1700000100, images:[{file:'000-123.png'}]}
  const [ls, list] = await api('/jobs')
  assert.equal(ls, 200)
  const listed = list.jobs.find(j => j.id === 'j-done')
  assert.equal(listed.kind, 'edit')
  assert.equal(listed.work_name, '作品')
  assert.equal(listed.created, '2023-11-14T22:13:20.000Z')

  const [gs, got] = await api('/jobs/j-done')
  assert.equal(gs, 200)
  assert.deepEqual(got.job.artifacts, [{index:0, name:'000-123.png', content_type:'image/png',
    url:'/automation/v1/jobs/j-done/artifacts/0'}])

  const file = await fetch(base + '/jobs/j-done/artifacts/0', {headers:AUTH})
  assert.equal(file.status, 200)
  assert.equal(file.headers.get('content-type'), 'image/png')
  assert.equal(await file.text(), 'png-bytes')
  assert.equal((await fetch(base + '/jobs/j-done/artifacts/0')).status, 401)

  assert.equal((await api('/jobs/nope'))[0], 404)
  const [cs] = await api('/jobs/j-done', {method:'DELETE'})
  assert.equal(cs, 200)
  assert.deepEqual(calls.at(-1), ['cancel', 'j-done'])
})
