// Owner-only API behind the "マンガ" sidebar panel: projects → pages (paneled) → panels.
// The name (ネーム) follows ja-manga-writing (skill/), written by an agent CLI; panel pictures come from
// native Anima (or P-H3 from character sheets) and are fixed with P-H3 edits. GPU work goes through the
// shared Anima/H3 queue; projects live in .runtime/manga/projects/<id>/project.json.
import {execFile} from 'node:child_process'
import {createHash, randomBytes} from 'node:crypto'
import {existsSync} from 'node:fs'
import {copyFile, mkdir, readdir, readFile, rename, rm, writeFile} from 'node:fs/promises'
import {basename, extname, join} from 'node:path'
import {fileURLToPath} from 'node:url'
import {promisify} from 'node:util'
import {agentNames, runCli} from '../../agents/cli.mjs'
import {cancelJob, enqueueGenerate, enqueueImage, jobView, listLoraNames, storePicture} from '../dsh-anima-lora-trainer/jobs.mjs'

const run = promisify(execFile)
const ROOT = fileURLToPath(new URL('../..', import.meta.url))
const HERE = fileURLToPath(new URL('.', import.meta.url))
const SKILL = join(HERE, 'skill')
const HELPER = join(HERE, 'manga.py')
const PROJECTS = join(ROOT, '.runtime/manga/projects')
const TRASH = join(ROOT, '.runtime/manga/trash')
const JOBS = join(ROOT, '.runtime/anima-native/jobs')
const PYTHON = process.env.DSH_MANGA_PYTHON || 'python3'
const SERVICES = ['claude', 'codex', 'google', 'grok', 'cursor', 'muse']
const FRAME = [70, 60, 930, 930]  // manga_name.py: basic frame (norm_1000) inside the margins
const DEFAULT_STYLE = 'black-and-white Japanese manga, clean ink line art, solid black fills and sparse screentone, plenty of white space, flat shading'
const DEFAULT_PREFIX = 'masterpiece, best quality, monochrome, greyscale, manga'
const ID = /^[0-9a-z-]{8,64}$/
const PANEL_ID = /^[\w.-]{1,40}$/

// ---------- projects ----------

const locks = new Map()
/** Runs fn(project) under the project's lock and saves what it returns true for (or always when save). */
function withProject(id, fn, save = true) {
  if (!ID.test(String(id))) return Promise.reject(Error('プロジェクトが見つかりません'))
  const previous = locks.get(id) ?? Promise.resolve()
  const next = previous.catch(() => {}).then(async () => {
    const project = await readProject(id)
    const result = await fn(project)
    if (save) await writeProject(project)
    return result
  })
  locks.set(id, next)
  next.finally(() => { if (locks.get(id) === next) locks.delete(id) }).catch(() => {})
  return next
}

async function readProject(id) {
  try { return JSON.parse(await readFile(join(PROJECTS, id, 'project.json'), 'utf8')) } catch { throw Error('プロジェクトが見つかりません') }
}

async function writeProject(project) {
  project.updated = new Date().toISOString()
  const file = join(PROJECTS, project.id, 'project.json')
  await writeFile(file + '.tmp', JSON.stringify(project, null, 1))
  await rename(file + '.tmp', file)
}

function text(value, label, max, required = false) {
  const s = typeof value === 'string' ? value.trim() : ''
  if (required && !s) throw Error(label + 'を入力してください')
  if (s.length > max) throw Error(label + 'は ' + max + ' 文字以内にしてください')
  return s
}

function int(value, label, min, max) {
  const n = Number(value)
  if (!Number.isInteger(n) || n < min || n > max) throw Error(label + 'は ' + min + '〜' + max + ' の整数にしてください')
  return n
}

async function listProjects() {
  const out = []
  for (const entry of await readdir(PROJECTS, {withFileTypes:true}).catch(() => [])) {
    if (!entry.isDirectory()) continue
    try {
      const p = JSON.parse(await readFile(join(PROJECTS, entry.name, 'project.json'), 'utf8'))
      const cover = firstAdopted(p)
      out.push({id:p.id, title:p.name.meta.title, idea:p.brief.idea, pages:p.name.pages?.length || 0, updated:p.updated,
                task:p.task?.state === 'running' ? p.task : undefined, cover})
    } catch {}
  }
  return out.sort((a, b) => String(b.updated).localeCompare(String(a.updated)))
}

function firstAdopted(p) {
  for (const page of p.name.pages || []) for (const panel of page.panels || []) {
    const pick = p.adopted?.[panel.id]
    if (pick) return pick.file
  }
  return null
}

async function createProject(body) {
  const title = text(body.title, '題名', 100, true)
  const idea = text(body.idea, '企画', 6000)
  const format = ['page', 'yonkoma', 'webtoon'].includes(body.format) ? body.format : 'page'
  const rating = body.rating === 'adult' ? 'adult' : 'general'
  const id = new Date().toISOString().slice(0, 10) + '-' + randomBytes(4).toString('hex')
  await mkdir(join(PROJECTS, id, 'images'), {recursive:true})
  const project = {
    id, created:new Date().toISOString(),
    brief:{idea, pages:int(body.pages ?? 8, 'ページ数', 1, 60), genre:text(body.genre, 'ジャンル', 200), format, rating,
           service:SERVICES.includes(body.service) ? body.service : 'claude'},
    render:{prefix:DEFAULT_PREFIX, negative:'', loras:[], sfx:'overlay', steps:34, cfg:4.5},
    name:{meta:{title, format, rating, first_page_side:'left', style:text(body.style, '画風', 2000) || DEFAULT_STYLE}, characters:[], pages:[]},
    takes:{}, adopted:{}, task:null,
  }
  await writeProject(project)
  return {id}
}

async function deleteProject(id) {
  if (!ID.test(String(id)) || !existsSync(join(PROJECTS, id))) throw Error('プロジェクトが見つかりません')
  await mkdir(TRASH, {recursive:true})
  await rename(join(PROJECTS, id), join(TRASH, id + '-' + Date.now()))
}

// ---------- analysis (ja-manga-writing scripts) ----------

const analyses = new Map()  // project id -> {hash, data}
async function analyze(project) {
  const doc = {...project.name, meta:{...project.name.meta}}
  const json = JSON.stringify(doc)
  const hash = createHash('sha1').update(json + project.render.sfx).digest('hex')
  const cached = analyses.get(project.id)
  if (cached?.hash === hash) return cached.data
  if (!doc.pages?.length) return {pages:[], issues:[], stats:{}}
  // The helper drops sound effects from the picture prompts when they are lettered on top (sfx = overlay).
  doc.meta.letter_sfx = project.render.sfx !== 'image'
  const file = join(PROJECTS, project.id, 'name.json')
  await writeFile(file, JSON.stringify(doc, null, 1))
  let data
  try {
    const {stdout} = await run(PYTHON, ['-I', HELPER, 'analyze', file], {maxBuffer:64 << 20, timeout:60000})
    data = JSON.parse(stdout)
  } catch (error) {
    data = {error:'ネームの解析に失敗しました: ' + String(error.stderr || error.message).slice(-500), pages:[], issues:[]}
  }
  analyses.set(project.id, {hash, data})
  return data
}

// ---------- layouts (コマ割りの型) ----------

let layoutCache
async function layouts() {
  layoutCache ??= JSON.parse(await readFile(join(SKILL, 'scripts/data/layouts.json'), 'utf8')).layouts
  return layoutCache
}

const SCENE = {talk:'会話', action:'動き', gag:'ギャグ', emotion:'感情', tense:'緊張'}
function layoutSummary(list) {
  return list.map(x => {
    const shots = (x.slots || []).map(s => s.shot).join(',')
    return `- ${x.id}（${x.boxes.length}コマ、${(x.scenes || []).map(s => SCENE[s] || s).join('・')}${x.evidence?.pages ? '、実例' + x.evidence.pages + 'p' : ''}）${x.name}。${x.purpose || ''}${shots ? ' shot目安: ' + shots + '。' : ''}${x.avoid ? ' 避ける: ' + x.avoid : ''}`
  }).join('\n')
}

function overlaps(a, b) {
  const o = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const area = x => (x[2] - x[0]) * (x[3] - x[1])
  return o > 0.15 * Math.min(area(a), area(b))
}

/** Lays the page's panels out on a layout type: boxes become bbox in the basic frame, in reading order. */
function applyLayout(page, lay) {
  const panels = page.panels || []
  if (lay.boxes.length !== panels.length) throw Error(`型 ${lay.id} は ${lay.boxes.length} コマです（このページは ${panels.length} コマ）`)
  const [fx1, fy1, fx2, fy2] = FRAME
  const area = b => (b[2] - b[0]) * (b[3] - b[1])
  panels.forEach((panel, k) => {
    const b = lay.boxes[k]
    for (const key of ['tier', 'width', 'column', 'height', 'inset']) delete panel[key]
    panel.bbox = [Math.round(fx1 + b[0] * (fx2 - fx1) / 1000), Math.round(fy1 + b[1] * (fy2 - fy1) / 1000),
                  Math.round(fx1 + b[2] * (fx2 - fx1) / 1000), Math.round(fy1 + b[3] * (fy2 - fy1) / 1000)]
    if (lay.boxes.some((o, j) => j !== k && overlaps(b, o) && area(b) < area(o))) panel.inset = true
  })
  delete page.tiers
  page.order = 'written'
  page.layout = lay.id
}

function clearLayout(page) {
  for (const panel of page.panels || []) delete panel.bbox, delete panel.inset
  delete page.order
  delete page.layout
}

// ---------- name writing with an agent CLI ----------

let skillText
async function skillContext() {
  if (skillText) return skillText
  const files = ['SKILL.md', 'references/name-format.md', 'references/story.md', 'references/paneling.md',
    'references/staging.md', 'references/dialogue-and-lettering.md', 'references/characters.md',
    'references/art-direction.md', 'references/genres.md']
  const parts = []
  for (const file of files) parts.push(`<file path="${file}">\n${await readFile(join(SKILL, file), 'utf8')}\n</file>`)
  const manpu = JSON.parse(await readFile(join(SKILL, 'scripts/data/manpu.json'), 'utf8'))
  const ids = group => Object.entries(manpu[group] || {}).map(([key, value]) => key + '=' + value.ja).join('、')
  parts.push(`<file path="漫符の語彙（manpu には marks、effects には effects の key を書く）">\nmarks: ${ids('marks')}\neffects: ${ids('effects')}\n</file>`)
  parts.push(`<file path="コマ割りの型（Manga109-s 全 87 冊から集計。pages[].layout に ID を書ける）">\n${layoutSummary(await layouts())}\n</file>`)
  skillText = parts.join('\n\n')
  return skillText
}

const OUTPUT_RULES = `## 出力の形
1. まず <plan> … </plan> に、作品の約束（何が面白いか・読者が何を楽しみに読むか）、ログライン、ページ配分、各ページの狙いとめくり、決めコマを短く書く。
2. 次に \`\`\`json … \`\`\` でネームを 1 つだけ出す。JSON は references/name-format.md の TOML と同じ構造（meta・characters・pages[].panels[]）。コメントは書けないので、意図は beat・note に書く。
3. 書き方の約束:
   - meta.style は全コマ共通の画風ブロック（英語）。依頼に画風があればそれを英語にして使う。
   - characters[].appearance は外見を英語で固定する（年齢感・髪型と髪色・目・服・体格・目印になる小物）。name は日本語、voice は口調。
   - コマの id は "P{ページ番号}-{そのページでの番号}"。visual・pose・expression・place_en・time_en は英語、beat・台詞・note は日本語。
   - pages[].layout に「コマ割りの型」の ID を書くと、そのページのコマ枠はその型になる（コマ数と読む順を型どおりにする。tiers・width は書かなくてよい）。型が合わないページは layout を書かずに tiers・width で割る。
   - 台詞は吹き出しに入る長さにする（1 吹き出しは短く、改行は ／）。擬音は type = "sfx"。
   - 面白さを最優先にする。掴み、ずらし、溜めと解放、めくりの引き、決めコマ（detail = "key"）を設計し、どのページも同じ段組み・顔の寄りの切り返しだけにしない。`

function briefText(project) {
  const b = project.brief, m = project.name.meta
  return `## 依頼
- 題名: ${m.title}
- 形式: ${b.format === 'yonkoma' ? '4コマ' : b.format === 'webtoon' ? '縦読み' : '横開き・右綴じ'}、${b.pages} ページ
- rating: ${b.rating}
- ジャンル: ${b.genre || '（おまかせ）'}
- 画風: ${m.style}
- 企画・アイデア:
${b.idea || '（おまかせ。面白い読切を考えてください）'}`
}

function extractName(answer) {
  const plan = /<plan>([\s\S]*?)<\/plan>/.exec(answer)?.[1]?.trim() || ''
  const fenced = [...answer.matchAll(/```(?:json)?\s*\n([\s\S]*?)\n```/g)].map(m => m[1])
  const candidates = fenced.length ? fenced : [answer.slice(answer.indexOf('{'), answer.lastIndexOf('}') + 1)]
  for (const candidate of candidates.reverse()) {
    try {
      const value = JSON.parse(candidate)
      if (value && typeof value === 'object') return {plan, value}
    } catch {}
  }
  throw Error('AI の返答から JSON のネームを読み取れませんでした')
}

/** Cleans the name an agent wrote: ids, page numbers, layout types → bbox. */
async function normalizeName(project, name) {
  if (!Array.isArray(name.pages) || !name.pages.length) throw Error('ネームにページがありません')
  const types = new Map((await layouts()).map(x => [x.id, x]))
  name.meta = {...project.name.meta, ...(name.meta || {}), format:project.brief.format, rating:project.brief.rating}
  name.characters = Array.isArray(name.characters) ? name.characters : []
  // Keep the LoRA and reference sheet the author attached to an existing character.
  const old = new Map((project.name.characters || []).map(c => [c.id, c]))
  for (const c of name.characters) {
    const before = old.get(c.id)
    if (before?.lora_name) c.lora_name = before.lora_name, c.lora_strength = before.lora_strength
    if (before?.ref) c.ref = before.ref
  }
  name.pages.forEach((page, i) => {
    page.number = i + 1
    page.panels = Array.isArray(page.panels) ? page.panels : []
    const seen = new Set()
    page.panels.forEach((panel, k) => {
      if (!PANEL_ID.test(String(panel.id)) || seen.has(panel.id)) panel.id = `P${i + 1}-${k + 1}`
      seen.add(panel.id)
    })
    const lay = types.get(page.layout)
    if (lay && lay.boxes.length === page.panels.length) applyLayout(page, lay)
    else delete page.layout
  })
  return name
}

const tasks = new Map()  // project id -> AbortController
async function startWrite(body) {
  const id = String(body.project)
  if (tasks.has(id)) throw Error('このプロジェクトでは AI が作業中です')
  const scope = body.scope === 'page' ? 'page' : 'all'
  const instruction = text(body.instruction, '指示', 4000)
  const service = SERVICES.includes(body.service) ? body.service : null
  let pageIndex = -1
  await withProject(id, project => {
    if (service) project.brief.service = service
    if (scope === 'page') {
      pageIndex = int(body.page, 'ページ', 1, project.name.pages.length) - 1
      if (!instruction) throw Error('ページの直し方を書いてください')
    }
    project.task = {state:'running', kind:scope === 'page' ? 'P' + (pageIndex + 1) + ' を直す' : project.name.pages.length ? 'ネームを書き直す' : 'ネームを切る',
                    service:project.brief.service, started:new Date().toISOString()}
  })
  const controller = new AbortController()
  tasks.set(id, controller)
  writeTask(id, scope, pageIndex, instruction, controller.signal)
    .catch(async error => withProject(id, project => {
      project.task = {...project.task, state:controller.signal.aborted ? 'cancelled' : 'error',
                      message:controller.signal.aborted ? '中止しました' : error.message, finished:new Date().toISOString()}
    }).catch(() => {}))
    .finally(() => tasks.delete(id))
  return {ok:true}
}

async function writeTask(id, scope, pageIndex, instruction, signal) {
  const project = await readProject(id)
  const context = await skillContext()
  const current = JSON.stringify(project.name, null, 1)
  let prompt
  if (scope === 'page') {
    prompt = `あなたは日本語の漫画のネームを切る作家です。以下のスキル（ja-manga-writing）の作法に従ってください。\n\n${context}\n\n${briefText(project)}\n\n## いまのネーム（JSON）\n${current}\n\n## 直すページ\nP${pageIndex + 1}（pages[${pageIndex}]）だけを、次の指示で直す。前後のページとのつながり・めくりを保つ。\n指示: ${instruction}\n\n## 出力の形\n<plan> に直す方針を短く書き、次に \`\`\`json で、直したページ 1 つ（pages[] の要素 1 つ。panels を含む）だけを出す。コマの id は "P${pageIndex + 1}-番号"。英語・日本語の使い分けとlayoutの書き方はいまのネームと同じ。`
  } else {
    const revise = project.name.pages.length
      ? `\n\n## いまのネーム（JSON。書き直しの元）\n${current}\n\n## 書き直しの指示\n${instruction || '面白さを上げる方向で全体を見直す'}` : instruction ? `\n\n## 追加の指示\n${instruction}` : ''
    prompt = `あなたは日本語の漫画のネームを切る作家です。以下のスキル（ja-manga-writing）の作法に従って、依頼の漫画のネームを切ってください。\n\n${context}\n\n${briefText(project)}${revise}\n\n${OUTPUT_RULES}`
  }
  const answer = await runCli(project.brief.service, prompt, AbortSignal.any([signal, AbortSignal.timeout(40 * 60000)]))
  const {plan, value} = extractName(answer)
  await withProject(id, async project => {
    if (scope === 'page') {
      if (!project.name.pages[pageIndex]) throw Error('ページが見つかりません')
      const name = structuredClone(project.name)
      name.pages[pageIndex] = Array.isArray(value.pages) ? value.pages[0] : value
      project.name = await normalizeName(project, name)
    } else {
      project.name = await normalizeName(project, value)
      if (plan) project.plan = plan
    }
    const data = await analyze(project)
    const fails = data.issues?.filter(i => i.level === 'FAIL').length || 0
    project.task = {...project.task, state:'done', finished:new Date().toISOString(),
                    message:(plan && scope === 'page' ? plan.slice(0, 400) + '\n' : '') + (data.error ? data.error : '検査: FAIL ' + fails + ' / WARN ' + (data.issues?.filter(i => i.level === 'WARN').length || 0))}
  })
}

function cancelWrite(id) {
  tasks.get(String(id))?.abort()
}

// ---------- panel pictures ----------

function findPanel(project, panelId) {
  for (const page of project.name.pages) for (const panel of page.panels || []) if (panel.id === panelId) return {page, panel}
  throw Error('コマが見つかりません: ' + panelId)
}

/** Picture size for a panel: its aspect in real units, about `side`² pixels, multiples of `step`. */
function panelSize(bbox, aspect, side, step, min, max) {
  const w = Math.max(1, bbox[2] - bbox[0]), h = Math.max(1, (bbox[3] - bbox[1]) * aspect)
  const ratio = Math.min(2.5, Math.max(0.4, w / h))
  const round = v => Math.min(max, Math.max(min, Math.round(v / step) * step))
  return [round(side * Math.sqrt(ratio)), round(side / Math.sqrt(ratio))]
}

async function uploadFile(path) {
  return storePicture(await readFile(path), extname(path).toLowerCase())
}

async function enqueuePanel(project, data, panelId, body) {
  const placed = data.pages.flatMap(page => page.panels.map(panel => ({...panel, aspect:page.aspect}))).find(p => p.id === panelId)
  if (!placed) throw Error('コマの割り付けがありません（ネームの検査エラーを直してください）: ' + panelId)
  const {panel} = findPanel(project, panelId)
  const r = project.render
  const extra = text(body.extra, '追加のプロンプト', 2000)
  const engine = body.engine === 'ph3' ? 'ph3' : 'anima'
  // Anima reads at most 512 T5 tokens: its prompt is the compact one, the avoid list goes to the negative.
  const base = [r.prefix, engine === 'anima' ? placed.anima : placed.prompt, panel.extra_prompt, extra].filter(Boolean).join(', ')
  const cast = new Map(project.name.characters.map(c => [c.id, c]))
  const inPanel = (panel.characters || []).map(c => cast.get(c.id)).filter(Boolean)
  const take = {id:randomBytes(6).toString('hex'), engine, state:'queued', files:[], created:new Date().toISOString(), prompt:base}
  if (engine === 'anima') {
    const loras = [...r.loras, ...inPanel.filter(c => c.lora_name).map(c => ({name:c.lora_name, strength:c.lora_strength ?? 1}))]
    const unique = [...new Map(loras.map(l => [l.name, l])).values()].slice(0, 4)
    const [width, height] = panelSize(placed.bbox, placed.aspect, 1024, 16, 512, 2048)
    const {id} = await enqueueGenerate({prompt:base, negative:[r.negative, placed.avoid].filter(Boolean).join(', '), width, height, loras:unique, count:int(body.count ?? 2, '枚数', 1, 8),
                                        steps:r.steps, cfg:r.cfg, seed:body.seed ?? null, fp8:true})
    Object.assign(take, {job:id, width, height, loras:unique})
  } else {
    // P-H3 from the characters' reference sheets: <Picture k> names each character.
    const refs = inPanel.filter(c => c.ref)
    const lines = refs.map((c, k) => `${c.name || c.id} is the character from <Picture ${k + 1}>`)
    const prompt = [...lines, base].join('. ')
    const images = []
    for (const c of refs) images.push(await uploadFile(join(PROJECTS, project.id, 'refs', c.ref)))
    const [width, height] = panelSize(placed.bbox, placed.aspect, 1536, 32, 512, 2912)
    const {id} = await enqueueImage({mode:images.length ? 'reference' : 't2i', prompt, images, width, height,
                                     count:int(body.count ?? 1, '枚数', 1, 8), seed:body.seed ?? null})
    Object.assign(take, {job:id, width, height, prompt})
  }
  ;(project.takes[panelId] ??= []).unshift(take)
  return take
}

async function generate(body) {
  return withProject(String(body.project), async project => {
    const data = await analyze(project)
    const ids = Array.isArray(body.panels) ? body.panels.map(String) : []
    if (!ids.length) throw Error('コマを選んでください')
    if (ids.length > 60) throw Error('一度に 60 コマまでです')
    for (const panelId of ids) await enqueuePanel(project, data, panelId, body)
    return {queued:ids.length}
  })
}

/** P-H3 edit of one picture of a take: the picture first, then the reference sheets of the panel's characters. */
async function edit(body) {
  return withProject(String(body.project), async project => {
    const panelId = String(body.panel)
    const {panel} = findPanel(project, panelId)
    const source = (project.takes[panelId] || []).find(t => t.id === body.take)
    const file = source?.files.find(f => f === body.file)
    if (!file) throw Error('直す絵が見つかりません')
    const instruction = text(body.instruction, '直し方', 3000, true)
    const images = [await uploadFile(join(PROJECTS, project.id, 'images', file))]
    const cast = new Map(project.name.characters.map(c => [c.id, c]))
    const lines = []
    if (body.refs !== false) for (const c of (panel.characters || []).map(c => cast.get(c.id)).filter(c => c?.ref)) {
      if (images.length >= 9) break
      images.push(await uploadFile(join(PROJECTS, project.id, 'refs', c.ref)))
      lines.push(`${c.name || c.id} looks like the character in <Picture ${images.length}>`)
    }
    const prompt = [instruction, ...lines].join('. ')
    const {id} = await enqueueImage({mode:'edit', prompt, images, width:0, height:0, count:int(body.count ?? 1, '枚数', 1, 4), seed:body.seed ?? null})
    const take = {id:randomBytes(6).toString('hex'), engine:'ph3-edit', job:id, state:'queued', files:[], created:new Date().toISOString(),
                  prompt, instruction, from:{take:source.id, file}}
    ;(project.takes[panelId] ??= []).unshift(take)
    return take
  })
}

/** Status of a queue job, also after it fell out of the queue's history (status.json and the output folder). */
async function jobState(jobId) {
  const view = await jobView(jobId)
  if (view) return {state:view.state, message:view.message, dir:view.request.output_dir, images:view.images || [], stage:view.stage, done:view.done}
  try {
    const request = JSON.parse(await readFile(join(JOBS, jobId, 'request.json'), 'utf8'))
    const status = JSON.parse(await readFile(join(JOBS, jobId, 'status.json'), 'utf8'))
    return {state:status.stage === 'done' ? 'done' : 'error', message:status.message || 'ジョブが見つかりません', dir:request.output_dir, images:status.images || []}
  } catch { return {state:'error', message:'ジョブが見つかりません'} }
}

/** Copies the pictures of finished jobs into the project; returns whether anything changed. */
async function settleTakes(project) {
  let changed = false
  for (const [panelId, takes] of Object.entries(project.takes)) for (const take of takes) {
    if (!['queued', 'running'].includes(take.state)) continue
    const job = await jobState(take.job)
    if (job.state === 'queued' || job.state === 'running') {
      if (take.state !== job.state) take.state = job.state, changed = true
      take.progress = job.state === 'running' ? (job.stage === 'generating' ? '生成中' : job.stage === 'loading' ? 'モデル読み込み' : '準備中') : undefined
      continue
    }
    if (job.state === 'done') {
      for (const image of job.images) {
        const name = `${panelId}-${take.id}-${basename(image.file)}`
        await copyFile(join(job.dir, image.file), join(PROJECTS, project.id, 'images', name)).catch(() => {})
        if (existsSync(join(PROJECTS, project.id, 'images', name))) take.files.push(name)
      }
      take.state = take.files.length ? 'done' : 'error'
      if (!take.files.length) take.message = '画像が見つかりません'
      // The first picture of a panel is adopted automatically.
      if (take.files.length && !project.adopted[panelId]) project.adopted[panelId] = {take:take.id, file:take.files[0]}
    } else {
      take.state = job.state === 'cancelled' ? 'cancelled' : 'error'
      take.message = job.message
    }
    delete take.progress
    changed = true
  }
  return changed
}

// Background sweep so finished pictures are kept even when nobody has the project open.
const pending = new Set()
async function sweep() {
  for (const id of [...pending]) {
    await withProject(id, async project => {
      await settleTakes(project)
      if (!Object.values(project.takes).flat().some(t => ['queued', 'running'].includes(t.state))) pending.delete(id)
    }).catch(() => pending.delete(id))
  }
}

async function getProject(id) {
  return withProject(String(id), async project => {
    if (await settleTakes(project)) await writeProject(project)
    if (Object.values(project.takes).flat().some(t => ['queued', 'running'].includes(t.state))) pending.add(project.id)
    return {project, analysis:await analyze(project), running:tasks.has(project.id)}
  }, false)
}

// ---------- edits from the panel ----------

const PANEL_KEYS = ['beat', 'visual', 'shot', 'angle', 'detail', 'background', 'place', 'place_en', 'time', 'time_en', 'transition',
  'note', 'characters', 'text', 'effects', 'bleed', 'extra_prompt']

async function updatePanel(body) {
  return withProject(String(body.project), project => {
    const {panel} = findPanel(project, String(body.panel))
    const patch = body.patch && typeof body.patch === 'object' ? body.patch : {}
    for (const key of PANEL_KEYS) if (key in patch) {
      if (patch[key] === '' || patch[key] === null) delete panel[key]
      else panel[key] = patch[key]
    }
    return {ok:true}
  })
}

async function updatePage(body) {
  return withProject(String(body.project), async project => {
    const page = project.name.pages[int(body.page, 'ページ', 1, project.name.pages.length) - 1]
    if ('purpose' in body) page.purpose = text(body.purpose, '狙い', 1000)
    if ('turn' in body) page.turn = text(body.turn, 'めくり', 1000)
    if (body.layout === '') clearLayout(page)
    else if (body.layout) {
      const lay = (await layouts()).find(x => x.id === body.layout)
      if (!lay) throw Error('型が見つかりません')
      applyLayout(page, lay)
    }
    if (body.addPanel) {
      const n = page.panels.length + 1
      let id = `P${page.number}-${n}`
      for (let k = n; page.panels.some(p => p.id === id); k++) id = `P${page.number}-${k + 1}`
      clearLayout(page)
      page.panels.push({id, beat:'', shot:'medium', ...(page.tiers ? {tier:page.tiers.length - 1} : {})})
    }
    if (body.removePanel) {
      const index = page.panels.findIndex(p => p.id === body.removePanel)
      if (index < 0) throw Error('コマが見つかりません')
      clearLayout(page)
      page.panels.splice(index, 1)
    }
    if (Array.isArray(body.tiers)) {
      clearLayout(page)
      page.tiers = body.tiers.map(Number).filter(v => v > 0).slice(0, 8)
    }
    if (body.panelTiers && typeof body.panelTiers === 'object') {
      for (const p of page.panels) if (p.id in body.panelTiers) {
        const v = body.panelTiers[p.id]
        p.tier = int(v.tier, '段', 0, 7)
        if (v.width) p.width = Number(v.width); else delete p.width
      }
    }
    return {ok:true}
  })
}

async function updateSettings(body) {
  return withProject(String(body.project), async project => {
    const m = project.name.meta, r = project.render
    if ('title' in body) m.title = text(body.title, '題名', 100, true)
    if ('style' in body) m.style = text(body.style, '画風', 2000) || DEFAULT_STYLE
    if ('idea' in body) project.brief.idea = text(body.idea, '企画', 6000)
    if ('genre' in body) project.brief.genre = text(body.genre, 'ジャンル', 200)
    if ('pages' in body) project.brief.pages = int(body.pages, 'ページ数', 1, 60)
    if ('prefix' in body) r.prefix = text(body.prefix, '先頭のプロンプト', 1000)
    if ('negative' in body) r.negative = text(body.negative, 'ネガティブプロンプト', 2000)
    if ('sfx' in body) r.sfx = body.sfx === 'image' ? 'image' : 'overlay'
    if ('steps' in body) r.steps = int(body.steps, 'step 数', 1, 100)
    if ('cfg' in body) { const v = Number(body.cfg); if (!(v >= 1 && v <= 20)) throw Error('CFG は 1〜20 です'); r.cfg = v }
    if ('loras' in body) {
      const available = new Set(await listLoraNames())
      r.loras = (Array.isArray(body.loras) ? body.loras : []).filter(l => l?.name).slice(0, 4).map(l => {
        if (!available.has(l.name)) throw Error('LoRA が見つかりません: ' + l.name)
        return {name:l.name, strength:Number(l.strength) || 1}
      })
    }
    return {ok:true}
  })
}

async function updateCharacter(body) {
  return withProject(String(body.project), async project => {
    const list = project.name.characters
    if (body.remove) { project.name.characters = list.filter(c => c.id !== body.remove); return {ok:true} }
    const c = body.character || {}
    const id = String(c.id || '').trim()
    if (!/^[a-z][\w-]{0,39}$/.test(id)) throw Error('人物の id は英小文字で始まる英数字にしてください')
    let target = list.find(x => x.id === (body.original || id))
    if (!target) list.push(target = {id})
    target.id = id
    target.name = text(c.name, '名前', 100)
    target.appearance = text(c.appearance, '外見', 2000)
    target.voice = text(c.voice, '口調', 500)
    if (c.lora_name) {
      if (!(await listLoraNames()).includes(c.lora_name)) throw Error('LoRA が見つかりません: ' + c.lora_name)
      target.lora_name = c.lora_name
      target.lora_strength = Number(c.lora_strength) || 1
    } else delete target.lora_name, delete target.lora_strength
    if (c.ref_data) {
      const ext = extname(String(c.ref_name || '')).toLowerCase()
      if (!['.png', '.jpg', '.jpeg', '.webp'].includes(ext)) throw Error('設定画は png・jpg・webp にしてください')
      const data = Buffer.from(String(c.ref_data), 'base64')
      if (!data.length || data.length > 20 << 20) throw Error('設定画は 20 MB までです')
      await mkdir(join(PROJECTS, project.id, 'refs'), {recursive:true})
      target.ref = id + '-' + randomBytes(4).toString('hex') + ext
      await writeFile(join(PROJECTS, project.id, 'refs', target.ref), data)
    }
    if (c.ref_from) {  // use a generated panel picture as the reference sheet
      const file = basename(String(c.ref_from))
      if (!existsSync(join(PROJECTS, project.id, 'images', file))) throw Error('絵が見つかりません')
      await mkdir(join(PROJECTS, project.id, 'refs'), {recursive:true})
      target.ref = id + '-' + randomBytes(4).toString('hex') + extname(file)
      await copyFile(join(PROJECTS, project.id, 'images', file), join(PROJECTS, project.id, 'refs', target.ref))
    }
    if (c.ref === null) delete target.ref
    return {ok:true}
  })
}

async function adopt(body) {
  return withProject(String(body.project), project => {
    const take = (project.takes[body.panel] || []).find(t => t.id === body.take)
    if (!take?.files.includes(body.file)) throw Error('絵が見つかりません')
    project.adopted[body.panel] = {take:take.id, file:body.file}
    return {ok:true}
  })
}

async function removeTake(body) {
  return withProject(String(body.project), async project => {
    const takes = project.takes[body.panel] || []
    const take = takes.find(t => t.id === body.take)
    if (!take) throw Error('見つかりません')
    if (['queued', 'running'].includes(take.state)) await cancelJob(take.job).catch(() => {})
    const files = body.file ? [body.file] : take.files
    take.files = take.files.filter(f => !files.includes(f))
    if (!take.files.length || !body.file) project.takes[body.panel] = takes.filter(t => t !== take)
    if (files.includes(project.adopted[body.panel]?.file)) delete project.adopted[body.panel]
    for (const f of files) await rm(join(PROJECTS, project.id, 'images', basename(f)), {force:true})
    return {ok:true}
  })
}

async function serveFile(params) {
  const id = params.get('project'), kind = params.get('kind'), file = basename(String(params.get('file')))
  if (!ID.test(String(id)) || !['images', 'refs'].includes(kind)) return null
  const ext = extname(file).slice(1).toLowerCase()
  if (!['png', 'jpg', 'jpeg', 'webp'].includes(ext)) return null
  const data = await readFile(join(PROJECTS, id, kind, file)).catch(() => null)
  return data && new Response(data, {headers:{'content-type':'image/' + (ext === 'jpg' ? 'jpeg' : ext), 'cache-control':'private, max-age=604800'}})
}

export function registerMangaApi(web) {
  web.effect(() => web.connection.fetch.register({
    path:'/api/darask.manga', methods:['GET', 'POST'], requestBody:'buffered',
    async fetch(request) {
      const reply = (data, status = 200) => Response.json(data, {status, headers:{'cache-control':'no-store'}})
      try {
        if (request.method === 'GET') {
          const params = new URL(request.url, 'http://local').searchParams
          if (params.has('file')) return await serveFile(params) ?? reply({error:'ファイルが見つかりません'}, 404)
          if (params.has('project')) return reply(await getProject(params.get('project')))
          if (params.has('layouts')) return reply({layouts:await layouts()})
          return reply({projects:await listProjects(), services:SERVICES.map(id => ({id, label:agentNames[id] || id})),
                        loras:await listLoraNames().catch(() => [])})
        }
        if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') return reply({error:'JSON が必要です'}, 415)
        const body = await request.json()
        const actions = {
          create:createProject, delete:b => deleteProject(b.project), write:startWrite, cancelWrite:b => cancelWrite(b.project),
          settings:updateSettings, character:updateCharacter, page:updatePage, panel:updatePanel,
          generate, edit, adopt, removeTake,
        }
        const action = actions[body.action]
        if (!action) return reply({error:'不明な操作です'}, 400)
        return reply(await action(body) ?? {ok:true})
      } catch (error) {
        return reply({error:error.message}, 400)
      }
    },
  }), 'Manga studio API')
  web.effect(() => {
    // A name being written when DSH stopped is reported as interrupted.
    (async () => {
      for (const entry of await readdir(PROJECTS, {withFileTypes:true}).catch(() => [])) {
        if (!entry.isDirectory() || !ID.test(entry.name)) continue
        await withProject(entry.name, project => {
          if (project.task?.state === 'running' && !tasks.has(project.id))
            project.task = {...project.task, state:'error', message:'DSH の再起動で中断しました。もう一度実行してください'}
          if (Object.values(project.takes).flat().some(t => ['queued', 'running'].includes(t.state))) pending.add(project.id)
        }).catch(() => {})
      }
    })()
    const timer = setInterval(() => sweep().catch(() => {}), 5000)
    return () => { clearInterval(timer); for (const c of tasks.values()) c.abort() }
  }, 'Manga studio sweep')
}
