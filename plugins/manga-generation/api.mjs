// Owner-only API behind the "マンガ" sidebar panel: projects → pages (paneled) → panels.
// The name (ネーム) follows ja-manga-writing (skill/), written by an agent CLI; panel pictures come from
// native Anima (or P-H3 from character sheets) and are fixed with P-H3 edits. GPU work goes through the
// shared Anima/H3 queue; projects live in outputs/manga/projects/<id>/project.json.
import {execFile, spawn} from 'node:child_process'
import {createHash, randomBytes} from 'node:crypto'
import {existsSync} from 'node:fs'
import {copyFile, mkdir, readdir, readFile, rename, rm, writeFile} from 'node:fs/promises'
import {basename, extname, join} from 'node:path'
import {fileURLToPath} from 'node:url'
import {promisify} from 'node:util'
import {agentNames, executable, runCli} from '../../harness/agents/cli.mjs'
import {cliOptions, optionArgs} from '../../harness/agents/options.mjs'
import {cancelJob, enqueueGenerate, enqueueImage, jobView, listLoraNames, storePicture} from '../media-shared/jobs.mjs'

const run = promisify(execFile)
const ROOT = fileURLToPath(new URL('../..', import.meta.url))
const HERE = fileURLToPath(new URL('.', import.meta.url))
const SKILL = join(HERE, 'skill')
const HELPER = join(HERE, 'manga.py')
const PROJECTS = join(ROOT, 'outputs/manga/projects')
const TRASH = join(ROOT, 'outputs/manga/trash')
const JOBS = join(ROOT, '.runtime/anima-native/jobs')
const PYTHON = process.env.DSH_MANGA_PYTHON || 'python3'
// typeset.py (lettering as SVG paths) needs fonttools + uharfbuzz: .runtime/manga-venv (see README)
const TYPESET_PYTHON = process.env.DSH_MANGA_TYPESET_PYTHON || join(ROOT, '.runtime/manga-venv/bin/python')
const TYPESET = join(SKILL, 'scripts/typeset.py')
// Fonts (SIL OFL) fetched with skill/scripts/fetch_fonts.py --dest .runtime/manga/fonts
const TYPESET_ENV = {...process.env, MANGA_FONT_DIR:process.env.MANGA_FONT_DIR || join(ROOT, '.runtime/manga/fonts')}
// Anima draws only people and places: lettering, balloons and the line effects come as layers on top.
const NO_TEXT = 'text, letters, speech bubble, sound effect lettering, signature'
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
  const loras = await checkLoras(body.loras)
  const id = new Date().toISOString().slice(0, 10) + '-' + randomBytes(4).toString('hex')
  await mkdir(join(PROJECTS, id, 'images'), {recursive:true})
  const project = {
    id, created:new Date().toISOString(),
    brief:{idea, pages:int(body.pages ?? 8, 'ページ数', 1, 60), genre:text(body.genre, 'ジャンル', 200), format, rating,
           service:SERVICES.includes(body.service) ? body.service : 'claude'},
    render:{prefix:DEFAULT_PREFIX, negative:'', loras, steps:34, cfg:4.5, engine:'ref', mono:true, restyle:true},
    places:{},
    name:{meta:{title, format, rating, first_page_side:'left', style:text(body.style, '画風', 2000) || DEFAULT_STYLE}, characters:[], pages:[]},
    takes:{}, adopted:{}, layers:{}, task:null,
  }
  await writeProject(project)
  return {id}
}

/** The work's style LoRAs (Anima, up to 4): applied to every Anima picture (panels, sheets, plates, style passes). */
async function checkLoras(list) {
  const available = new Set(await listLoraNames())
  return (Array.isArray(list) ? list : []).filter(l => l?.name).slice(0, 4).map(l => {
    if (!available.has(l.name)) throw Error('LoRA が見つかりません: ' + l.name)
    const strength = Number(l.strength)
    if (!Number.isFinite(strength) || strength < -2 || strength > 2) throw Error('LoRA の強さは -2〜2 にしてください')
    return {name:l.name, strength}
  })
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
  const hash = createHash('sha1').update(json + 'overlay-v3').digest('hex')
  const cached = analyses.get(project.id)
  if (cached?.hash === hash) return cached.data
  if (!doc.pages?.length) return {pages:[], issues:[], stats:{}}
  // Text, balloons and the effects/marks drawn as layers stay out of the picture prompts (skill/scripts/name_layers.py).
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
  return withProject(String(id), project => { if (project.qaLoop) project.qaLoop.waiting = [] })
}

// ---------- panel pictures ----------

function findPanel(project, panelId) {
  for (const page of project.name.pages) for (const panel of page.panels || []) if (panel.id === panelId) return {page, panel}
  throw Error('コマが見つかりません: ' + panelId)
}

// Anima reads at most 512 T5 tokens (about 3.5 English characters each); keep a margin.
// Same rule as skill/scripts/name_layers.py (ANIMA_BUDGET, fit_prompt).
const ANIMA_BUDGET = 1650
/** The Anima prompt with the added parts kept whole: the panel prompt loses its last sentences until everything fits. */
function fitAnima(prefix, prompt, extras) {
  const tail = extras.filter(Boolean).join(', ')
  let body = String(prompt || '')
  const room = ANIMA_BUDGET - (prefix || '').length - tail.length - 4
  while (body.length > room && body.includes('. ')) body = body.slice(0, body.lastIndexOf('. ', body.length - 2) + 1)
  if (body.length > room) body = body.slice(0, Math.max(0, room))
  return [prefix, body, tail].filter(Boolean).join(', ')
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
  const engine = ['anima', 'ph3', 'ref'].includes(body.engine) ? body.engine : r.engine || 'ref'
  // Anima reads at most 512 T5 tokens: its prompt is the compact one, the avoid list goes to the negative.
  const base = engine === 'anima' ? fitAnima(r.prefix, placed.anima, [panel.extra_prompt, extra])
    : [r.prefix, placed.prompt, panel.extra_prompt, extra].filter(Boolean).join(', ')
  const cast = new Map(project.name.characters.map(c => [c.id, c]))
  const inPanel = (panel.characters || []).map(c => cast.get(c.id)).filter(Boolean)
  const take = {id:randomBytes(6).toString('hex'), engine, state:'queued', files:[], created:new Date().toISOString(), prompt:base}
  if (engine === 'anima') {
    const loras = [...r.loras, ...inPanel.filter(c => c.lora_name).map(c => ({name:c.lora_name, strength:c.lora_strength ?? 1}))]
    const unique = [...new Map(loras.map(l => [l.name, l])).values()].slice(0, 4)
    const [width, height] = panelSize(placed.bbox, placed.aspect, 1024, 16, 512, 2048)
    const {id} = await enqueueGenerate({prompt:base, negative:[r.negative, placed.avoid, NO_TEXT].filter(Boolean).join(', '), width, height, loras:unique, count:int(body.count ?? 2, '枚数', 1, 8),
                                        steps:r.steps, cfg:r.cfg, seed:body.seed ?? null, fp8:true})
    Object.assign(take, {job:id, width, height, loras:unique})
  } else if (engine === 'ref') {
    // Anima draws the style (turnaround sheets, place plates); P-H3 draws the panel from them as told.
    const {prompt, images} = await referencePrompt(project, placed, panel, [panel.extra_prompt, extra])
    const [width, height] = panelSize(placed.bbox, placed.aspect, 1344, 32, 512, 2912)
    const {id} = await enqueueImage({mode:images.length ? 'reference' : 't2i', prompt, images, width, height,
                                     count:int(body.count ?? 2, '枚数', 1, 8), seed:body.seed ?? null})
    Object.assign(take, {job:id, width, height, prompt, refs:images.length})
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
  if (body.autoAdopt) take.autoAdopt = true
  ;(project.takes[panelId] ??= []).unshift(take)
  pending.add(project.id)
  return take
}

/** The P-H3 prompt and pictures for "draw with references": each character's turnaround sheet, the place's plate,
 *  then the panel's camera, positions, poses and expressions from the name. */
async function referencePrompt(project, placed, panel, extras) {
  const sc = placed.scene || {}
  const cast = new Map(project.name.characters.map(c => [c.id, c]))
  const images = [], lines = []
  const people = []
  for (const ch of sc.characters || []) {
    const c = cast.get(ch.id)
    const name = ch.name || ch.id
    if (c?.ref && images.length < 8) {
      images.push(await uploadFile(join(PROJECTS, project.id, 'refs', c.ref)))
      lines.push(`<Picture ${images.length}> is the character sheet of ${name} (one person, possibly shown from several angles; draw ${name} only once).`)
    } else if (c?.appearance) lines.push(`${name}: ${c.appearance}.`)
    people.push(ch)
  }
  const plate = sc.plate && sc.place_key ? Object.values(project.places || {}).find(p => p.key === sc.place_key && p.ref) : null
  if (plate) {
    images.push(await uploadFile(join(PROJECTS, project.id, 'refs', plate.ref)))
    lines.push(`<Picture ${images.length}> is the place (${sc.place}); keep its layout, furniture and objects, seen from this panel's camera.`)
  } else if (sc.background) lines.push(sc.background.charAt(0).toUpperCase() + sc.background.slice(1) + '.')
  else if (sc.place) lines.push(`Place: ${sc.place}.`)
  const out = [
    `Draw one manga panel in exactly the same black-and-white ink style as the reference pictures: ${project.name.meta.style}.`,
    ...lines,
    sc.camera ? `Camera: ${sc.camera}.` : '',
    people.length ? `Exactly ${people.length} ${people.length > 1 ? 'people' : 'person'}.` : 'No people.',
    ...people.map(ch => `${ch.name}${ch.position ? ' ' + ch.position : ''}: ${[ch.pose, ch.outfit && 'wearing ' + ch.outfit].filter(Boolean).join(', ')}` +
      (ch.expression ? `; facial expression clearly readable and strong: ${ch.expression}` : '') + '.'),
    sc.visual ? `Scene: ${sc.visual}.` : '',
    sc.detail === 'key' ? 'This is a key panel: draw the characters and the place with full care and line work.' : sc.detail === 'light' ? 'Keep the drawing simple with few lines.' : '',
    ...extras.filter(Boolean),
    'No text, no letters, no speech bubbles, no sound effects, no effect lines, no color.',
  ]
  return {prompt:out.filter(Boolean).join(' '), images}
}

/** Anima edit of one picture of a panel: "style" re-draws the whole picture lightly with the panel's Anima prompt
 *  and LoRAs (the work's style over a P-H3 picture), "region" re-draws only inside the painted mask. */
async function animaEditPanel(project, panelId, body, data) {
  const {panel} = findPanel(project, panelId)
  const source = (project.takes[panelId] || []).find(t => t.id === body.take)
  const file = source?.files.find(f => f === body.file)
  if (!file) throw Error('直す絵が見つかりません')
  const mode = body.mode === 'region' ? 'region' : 'style'
  const r = project.render
  data ??= await analyze(project)
  const placed = data.pages.flatMap(page => page.panels).find(p => p.id === panelId)
  const cast = new Map(project.name.characters.map(c => [c.id, c]))
  const inPanel = (panel.characters || []).map(c => cast.get(c.id)).filter(Boolean)
  const loras = [...r.loras, ...inPanel.filter(c => c.lora_name).map(c => ({name:c.lora_name, strength:c.lora_strength ?? 1}))]
  const unique = [...new Map(loras.map(l => [l.name, l])).values()].slice(0, 4)
  let prompt, mask
  if (mode === 'region') {
    const told = text(body.prompt, '描き直す内容', 1500, true)
    prompt = fitAnima(r.prefix, project.name.meta.style, [told])
    if (!body.mask) throw Error('描き直す範囲を塗ってください')
    mask = await storePicture(Buffer.from(String(body.mask), 'base64'), '.png')
  } else {
    prompt = fitAnima(r.prefix, placed?.anima || '', [panel.extra_prompt, text(body.prompt, '追加のプロンプト', 1000)])
  }
  const denoise = Number(body.denoise ?? (mode === 'region' ? 0.9 : 0.4))
  const {id} = await enqueueGenerate({prompt, negative:[r.negative, placed?.avoid, NO_TEXT, 'color'].filter(Boolean).join(', '),
    image:await uploadFile(join(PROJECTS, project.id, 'images', file)), ...(mask ? {mask} : {}), denoise,
    mask_grow:int(body.grow ?? 8, 'マスクの広げ', 0, 64), mask_feather:int(body.feather ?? 12, 'マスクのぼかし', 0, 64),
    width:0, height:0, loras:unique, count:int(body.count ?? 2, '枚数', 1, 8), steps:r.steps, cfg:r.cfg, seed:body.seed ?? null, fp8:true})
  const take = {id:randomBytes(6).toString('hex'), engine:'anima-edit', mode, job:id, state:'queued', files:[], created:new Date().toISOString(),
                prompt, denoise, from:{take:source.id, file}, ...(body.instruction ? {instruction:String(body.instruction).slice(0, 300)} : {}),
                ...(body.autoAdopt ? {autoAdopt:true} : {})}
  ;(project.takes[panelId] ??= []).unshift(take)
  pending.add(project.id)
  return take
}

async function animaEdit(body) {
  return withProject(String(body.project), async project => {
    const take = await animaEditPanel(project, String(body.panel), body)
    return {id:take.id}
  })
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
  return withProject(String(body.project), project => editPanel(project, String(body.panel), body))
}

/** Queues a P-H3 edit of one picture of a panel (the picture first, then the characters' reference sheets). */
async function editPanel(project, panelId, body) {
  {
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
                  prompt, instruction, from:{take:source.id, file}, ...(body.autoAdopt ? {autoAdopt:true} : {})}
    ;(project.takes[panelId] ??= []).unshift(take)
    pending.add(project.id)
    return take
  }
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
      const mono = take.engine !== 'anima' && project.render.mono !== false
      for (const image of job.images) {
        const name = `${panelId}-${take.id}-${basename(image.file)}`
        const from = join(job.dir, image.file), to = join(PROJECTS, project.id, 'images', name)
        // P-H3 can bring colour in from the references: keep the page black and white.
        if (mono) await run(PYTHON, ['-I', HELPER, 'mono', from, to], {timeout:60000}).catch(() => copyFile(from, to)).catch(() => {})
        else await copyFile(from, to).catch(() => {})
        if (existsSync(join(PROJECTS, project.id, 'images', name))) take.files.push(name)
      }
      take.state = take.files.length ? 'done' : 'error'
      if (!take.files.length) take.message = '画像が見つかりません'
      if (panelId.startsWith('@place:')) {
        const place = project.places?.[panelId.slice(7)]
        if (place && !place.ref && take.files.length) {
          await mkdir(join(PROJECTS, project.id, 'refs'), {recursive:true})
          place.ref = 'place-' + panelId.slice(7) + '-' + randomBytes(3).toString('hex') + extname(take.files[0])
          await copyFile(join(PROJECTS, project.id, 'images', take.files[0]), join(PROJECTS, project.id, 'refs', place.ref))
        }
      } else if (panelId.startsWith('@char:')) {
        // A character sheet: the first one becomes the character's reference when it has none.
        const c = project.name.characters.find(c => c.id === panelId.slice(6))
        if (c && !c.ref && take.files.length) {
          await mkdir(join(PROJECTS, project.id, 'refs'), {recursive:true})
          c.ref = c.id + '-' + randomBytes(4).toString('hex') + extname(take.files[0])
          await copyFile(join(PROJECTS, project.id, 'images', take.files[0]), join(PROJECTS, project.id, 'refs', c.ref))
        }
      } else if (take.files.length && (!project.adopted[panelId] || take.autoAdopt)) {
        // The first picture of a panel is adopted; so is a fix the inspection asked for.
        project.adopted[panelId] = {take:take.id, file:take.files[0]}
      }
      // P-H3 pictures (drawn with references, or fixed with a P-H3 edit) then Anima over them: the work's style and
      // LoRAs back on the picture, so the page keeps one style.
      if (['ref', 'ph3', 'ph3-edit'].includes(take.engine) && take.files.length && project.render.restyle !== false && !panelId.startsWith('@')) {
        try {
          const follow = await animaEditPanel(project, panelId, {take:take.id, file:take.files[0], mode:'style', count:1,
            autoAdopt:project.adopted[panelId]?.take === take.id})
          follow.restyleOf = take.id
        } catch (error) { take.message = 'Anima の画風そろえを入れられませんでした: ' + error.message }
      }
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
    await continueInspection(id).catch(() => {})
  }
}

async function getProject(id) {
  return withProject(String(id), async project => {
    if (await settleTakes(project)) await writeProject(project)
    if (Object.values(project.takes).flat().some(t => ['queued', 'running'].includes(t.state))) pending.add(project.id)
    const analysis = await analyze(project)
    analysis.places = (analysis.places || []).map(p => ({...p, id:placeId(p.key)}))
    return {project, analysis, running:tasks.has(project.id)}
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
    if ('engine' in body) r.engine = ['ref', 'anima', 'ph3'].includes(body.engine) ? body.engine : 'ref'
    if ('mono' in body) r.mono = body.mono !== false
    if ('restyle' in body) r.restyle = body.restyle !== false
    if ('steps' in body) r.steps = int(body.steps, 'step 数', 1, 100)
    if ('cfg' in body) { const v = Number(body.cfg); if (!(v >= 1 && v <= 20)) throw Error('CFG は 1〜20 です'); r.cfg = v }
    if ('loras' in body) r.loras = await checkLoras(body.loras)
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

// ---------- lettering and effects layers (写植・演出) ----------

let catalog
/** Sound-effect styles and the effect/mark names typeset.py draws (cached). */
async function typesetCatalog() {
  catalog ??= run(TYPESET_PYTHON, ['-I', TYPESET, 'styles'], {timeout:60000, env:TYPESET_ENV}).then(({stdout}) => JSON.parse(stdout))
    .catch(error => { catalog = undefined; throw Error('写植の準備ができていません（.runtime/manga-venv と fonts）: ' + error.message) })
  return catalog
}

let worker = null
const waiting = new Map()
/** The resident typesetter: one JSON line per request. */
function typeset(panels) {
  if (!worker) {
    worker = spawn(TYPESET_PYTHON, ['-I', TYPESET, 'serve'], {cwd:HERE, env:TYPESET_ENV, stdio:['pipe', 'pipe', 'pipe']})
    let buffer = ''
    worker.stdout.on('data', chunk => {
      buffer += chunk
      let end
      while ((end = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1)
        try { const res = JSON.parse(line); waiting.get(res.id)?.resolve(res); waiting.delete(res.id) } catch {}
      }
    })
    worker.stderr.on('data', chunk => process.stderr.write('[manga typeset] ' + chunk))
    const fail = () => { worker = null; for (const w of waiting.values()) w.reject(Error('写植の処理が止まりました')); waiting.clear() }
    worker.on('exit', fail)
    worker.on('error', fail)
  }
  const id = randomBytes(6).toString('hex')
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { waiting.delete(id); reject(Error('写植が時間内に終わりませんでした')) }, 120000)
    waiting.set(id, {resolve:res => { clearTimeout(timer); res.error ? reject(Error(res.error)) : resolve(res.panels) }, reject})
    worker.stdin.write(JSON.stringify({id, panels}) + '\n')
  })
}

const rendered = new Map()  // hash -> {svg, boxes}
/** SVG overlays of a page's panels: stored layers, or the automatic ones from the name. */
async function overlays(project, data, pageNumber) {
  const page = data.pages.find(p => p.number === pageNumber)
  if (!page) throw Error('ページが見つかりません')
  const items = page.panels.map(panel => {
    const layers = project.layers?.[panel.id]?.items ?? panel.layers
    const req = {aspect:panel.aspect, layers, seed:project.id + '/' + panel.id}
    return {panel, req, hash:createHash('sha1').update(JSON.stringify(req)).digest('hex')}
  })
  const missing = items.filter(item => !rendered.has(item.hash))
  if (missing.length) {
    const out = await typeset(missing.map(item => item.req))
    missing.forEach((item, i) => rendered.set(item.hash, out[i]))
    if (rendered.size > 3000) rendered.delete(rendered.keys().next().value)
  }
  return Object.fromEntries(items.map(item => [item.panel.id, {...rendered.get(item.hash),
    source:project.layers?.[item.panel.id]?.source || 'auto', layers:item.req.layers}]))
}

const LAYER_TYPES = ['balloon', 'sfx', 'text', 'effect', 'mark']
const BALLOON_STYLES = ['speech', 'shout', 'whisper', 'thought', 'narration', 'monologue']
const clamp = (v, lo, hi, d) => { const n = Number(v); return Number.isFinite(n) ? Math.min(hi, Math.max(lo, Math.round(n * 1000) / 1000)) : d }

/** Keeps only known layer fields with sane values. */
async function cleanLayers(items) {
  const {styles, effects, marks} = await typesetCatalog()
  const styleKeys = new Set(styles.map(s => s.key))
  if (!Array.isArray(items)) throw Error('層の形が正しくありません')
  return items.slice(0, 60).filter(l => l && LAYER_TYPES.includes(l.type)).map(l => {
    const t = String(l.text ?? '').slice(0, 200)
    switch (l.type) {
      case 'balloon': {
        const lines = (Array.isArray(l.lines) ? l.lines : String(l.text ?? '').split(/[／/]/)).map(x => String(x).slice(0, 40)).filter(Boolean).slice(0, 12)
        const out = {type:'balloon', style:BALLOON_STYLES.includes(l.style) ? l.style : 'speech', lines,
                     x:clamp(l.x, -0.15, 1, 0.5), y:clamp(l.y, -0.15, 1, 0.05), w:clamp(l.w, 0.03, 1.2, 0.2), h:clamp(l.h, 0.03, 1.2, 0.3)}
        if (Array.isArray(l.tail) && l.tail.length === 2) out.tail = [clamp(l.tail[0], -0.5, 1.5, 0.5), clamp(l.tail[1], -0.5, 1.5, 0.5)]
        if (l.size !== undefined) out.size = clamp(l.size, 0.3, 2, 1)
        return out
      }
      case 'sfx': return {type:'sfx', text:t.slice(0, 24), style:styleKeys.has(l.style) ? l.style : 'impact', x:clamp(l.x, -0.15, 1, 0.1), y:clamp(l.y, -0.15, 1, 0.1),
        size:clamp(l.size, 0.02, 0.8, 0.14), vertical:l.vertical !== false, rotate:clamp(l.rotate, -90, 90, 0),
        ...(l.color === 'white' || l.color === 'black' ? {color:l.color} : {})}
      case 'text': return {type:'text', text:t, x:clamp(l.x, -0.15, 1, 0.1), y:clamp(l.y, -0.15, 1, 0.1), size:clamp(l.size, 0.01, 0.3, 0.04),
        vertical:!!l.vertical, ...(styleKeys.has(l.style) ? {style:l.style} : {})}
      case 'effect': return effects.includes(l.name) ? {type:'effect', name:l.name, cx:clamp(l.cx, -0.5, 1.5, 0.5), cy:clamp(l.cy, -0.5, 1.5, 0.5),
        r:clamp(l.r, 0, 1.5, 0.28), angle:clamp(l.angle, -360, 360, 0), density:clamp(l.density, 0.2, 3, 1)} : null
      case 'mark': return marks.includes(l.name) ? {type:'mark', name:l.name, x:clamp(l.x, -0.15, 1.1, 0.5), y:clamp(l.y, -0.15, 1.1, 0.1),
        size:clamp(l.size, 0.02, 0.5, 0.08), rotate:clamp(l.rotate, -180, 180, 0)} : null
    }
  }).filter(Boolean)
}

async function saveLayers(body) {
  const items = body.reset ? null : await cleanLayers(body.items)
  return withProject(String(body.project), project => {
    findPanel(project, String(body.panel))
    project.layers ??= {}
    if (items === null) delete project.layers[body.panel]
    else project.layers[body.panel] = {items, source:'edited', at:new Date().toISOString()}
    return {ok:true}
  })
}

let letteringRules
async function letteringContext() {
  if (letteringRules) return letteringRules
  const {styles, effects, marks} = await typesetCatalog()
  const manpu = JSON.parse(await readFile(join(SKILL, 'scripts/data/manpu.json'), 'utf8'))
  const ja = (group, key) => manpu[group]?.[key]?.ja || manpu.marks?.[key]?.ja || manpu.effects?.[key]?.ja || key
  letteringRules = [
    `<file path="references/dialogue-and-lettering.md">\n${await readFile(join(SKILL, 'references/dialogue-and-lettering.md'), 'utf8')}\n</file>`,
    '<file path="書き文字のスタイル（sfx の style）">\n' + styles.map(s => `- ${s.key}: ${s.ja}。${s.use}`).join('\n') + '\n</file>',
    '<file path="層で描ける効果（effect の name）">\n' + effects.map(k => `${k}=${ja('effects', k)}`).join('、') + '\n</file>',
    '<file path="層で描ける漫符（mark の name）">\n' + marks.map(k => `${k}=${ja('marks', k)}`).join('、') + '\n</file>',
  ].join('\n\n')
  return letteringRules
}

const LAYER_SCHEMA = `## 層の書き方（座標はコマの中の割合。x は幅、y は高さに対して 0〜1。左上が 0,0）
- 吹き出し: {"type":"balloon","style":"speech|shout|whisper|thought|narration|monologue","lines":["右の列","次の列"],"x":左上x,"y":左上y,"w":幅,"h":高さ,"tail":[話者の口元のx,y]}
  縦書き。lines は右の列から。1 列はおよそ 4〜9 字。文字の大きさは枠に合わせて自動で決まるので、w・h を字数に合う大きさにする（目安: 1 列 6 字・2 列なら w 0.12〜0.2、h 0.3〜0.45 をコマの縦横比で調整）。
  ナレーション・モノローグは tail なしの四角。叫びは shout（トゲ）、心の声は thought（雲）、ささやきは whisper（点線）。
- 書き文字（擬音）: {"type":"sfx","text":"ドンッ","style":"スタイルの key","x":1 字目の左上x,"y":y,"size":字の高さ（コマの幅に対する比。小さな物音 0.05〜0.08、普通 0.1〜0.16、決め 0.2〜0.35）,"vertical":true|false,"rotate":角度(-45〜45),"color":"black|white"（省略可。ベタの上は white）}
- 作中の文字・欄外: {"type":"text","text":"文字（改行は／）","x":x,"y":y,"size":0.03〜0.06,"vertical":true|false}
- 効果: {"type":"effect","name":"効果の name","cx":中心x,"cy":中心y,"r":中心の空ける半径（コマの幅に対する比）,"angle":向き（度。スピード線の流れ）,"density":0.5〜2}
- 漫符: {"type":"mark","name":"漫符の name","x":x,"y":y,"size":0.05〜0.12,"rotate":角度}`

const LETTERING_RULES = `## 写植・演出の約束
1. 台詞の文字はネームのとおり（句読点・…・！？を含む）。列の分け方は変えてよい。台詞を足したり言い換えたりしない。ネームの台詞は全部どこかの吹き出しに入れる。
2. 吹き出しは、顔・目・手元・決めの動作・見せたい物に重ねない。コマの中の読む順（右上 → 左下、右綴じ）に、話す順で並べる。話者の近くに置き、しっぽの先は話者の口元。枠越えは x・y が -0.1 まで。
3. 書き文字は絵の動きと音の出どころに沿わせ、場面に合うスタイルを選ぶ。人物の顔にはかけない。決めのコマでは大きく、角度をつけてよい。小さな物音は小さく。
4. 演出（集中線・スピード線・フラッシュなど）は、ネームの effects と、場面の beat を強めるときだけ。集中線・フラッシュの中心は見せたいもの、r は主題（顔・手）を覆わない大きさ。何もかもに線を足さない。
5. 漫符はネームの manpu を、その人物の頭の横・上に。感情を言い当てる程度に小さく。
6. 絵が無いコマは、ネームと人物の位置（右・中央・左）から推定して置く。`

/** Starts the AI lettering of a page (or some of its panels): it looks at each panel picture and places the layers. */
async function startLetter(body) {
  const id = String(body.project)
  if (tasks.has(id)) throw Error('このプロジェクトでは AI が作業中です')
  const instruction = text(body.instruction, '指示', 3000)
  const service = SERVICES.includes(body.service) ? body.service : null
  let pageNumber, only
  await withProject(id, project => {
    pageNumber = int(body.page, 'ページ', 1, project.name.pages.length)
    only = Array.isArray(body.panels) && body.panels.length ? body.panels.map(String) : null
    project.task = {state:'running', kind:'P' + pageNumber + (only ? ' ' + only.join('・') : '') + ' の写植・演出',
                    service:service || project.brief.service, started:new Date().toISOString()}
  })
  const controller = new AbortController()
  tasks.set(id, controller)
  letterTask(id, pageNumber, only, instruction, service, controller.signal)
    .catch(async error => withProject(id, project => {
      project.task = {...project.task, state:controller.signal.aborted ? 'cancelled' : 'error',
                      message:controller.signal.aborted ? '中止しました' : error.message, finished:new Date().toISOString()}
    }).catch(() => {}))
    .finally(() => tasks.delete(id))
  return {ok:true}
}

async function letterTask(id, pageNumber, only, instruction, chosen, signal) {
  const project = await readProject(id)
  const service = chosen || project.brief.service
  const data = await analyze(project)
  const page = data.pages.find(p => p.number === pageNumber)
  const name = project.name.pages.find(p => p.number === pageNumber)
  const panels = page.panels.filter(p => !only || only.includes(p.id))
  if (!panels.length) throw Error('コマが見つかりません')
  const crops = join(PROJECTS, id, 'crops')
  await rm(crops, {recursive:true, force:true})
  await mkdir(crops, {recursive:true})
  const cast = new Map(project.name.characters.map(c => [c.id, c]))
  const vision = service === 'claude'
  const blocks = []
  for (const placed of panels) {
    const panel = name.panels.find(p => p.id === placed.id)
    const file = project.adopted?.[placed.id]?.file
    let picture = null
    if (file && vision) {
      picture = placed.id.replace(/[^\w.-]/g, '_') + '.png'
      await run(PYTHON, ['-I', HELPER, 'crop', join(PROJECTS, id, 'images', file), join(crops, picture), String(placed.aspect)], {timeout:60000})
    }
    blocks.push({
      id:placed.id, aspect_w_over_h:placed.aspect, picture:picture ? './' + picture : null,
      beat:panel.beat, shot:panel.shot, angle:panel.angle, detail:panel.detail,
      characters:(panel.characters || []).map(c => ({id:c.id, name:cast.get(c.id)?.name, pos:c.pos, pose:c.pose, expression:c.expression, manpu:c.manpu})),
      text:(panel.text || []).map(t => ({type:t.type || 'dialogue', speaker:t.speaker ? cast.get(t.speaker)?.name || t.speaker : undefined,
                                         text:Array.isArray(t.lines) ? t.lines.join('／') : t.text})),
      effects:panel.effects, current_layers:project.layers?.[placed.id]?.items ?? placed.layers,
    })
  }
  const prompt = `あなたは日本語の漫画の写植と仕上げ（吹き出し・台詞・書き文字・効果線・漫符）を担当します。絵は Anima が人物と背景だけを描いてあり、文字と演出はあなたが層として載せます。` +
    (vision ? '\n各コマの picture（このフォルダの画像。コマの枠と同じ縦横比に切り抜き済み）を Read で必ず見て、顔・手・見せたい物の位置を確かめてから置いてください。' : '\n（このモデルでは絵を見られないので、ネームと人物の位置から推定して置いてください。）') +
    `\n\n${await letteringContext()}\n\n${LAYER_SCHEMA}\n\n${LETTERING_RULES}\n\n## このページ\nP${pageNumber}。狙い: ${name.purpose || '—'}\n` +
    (instruction ? `\n## 作者の指示（最優先）\n${instruction}\n` : '') +
    `\n## コマ（読む順）\n${JSON.stringify(blocks, null, 1)}\n\n## 出力\n最後に \`\`\`json で {"panels":[{"id":"コマの id","layers":[...]}],"note":"置き方の要点を 1〜3 行（日本語）"} を 1 つだけ。current_layers は叩き台で、良ければ流用してよい。`
  const answer = vision ? await runVision(prompt, crops, signal) : await runCli(service, prompt, AbortSignal.any([signal, AbortSignal.timeout(20 * 60000)]))
  const fenced = [...answer.matchAll(/```(?:json)?\s*\n([\s\S]*?)\n```/g)].map(m => m[1])
  let value
  for (const candidate of (fenced.length ? fenced : [answer.slice(answer.indexOf('{'), answer.lastIndexOf('}') + 1)]).reverse()) {
    try { value = JSON.parse(candidate); break } catch {}
  }
  if (!Array.isArray(value?.panels)) throw Error('AI の返答から層を読み取れませんでした')
  const cleaned = []
  for (const item of value.panels) if (panels.some(p => p.id === item?.id)) cleaned.push([item.id, await cleanLayers(item.layers || [])])
  await withProject(id, project => {
    project.layers ??= {}
    for (const [panelId, items] of cleaned) project.layers[panelId] = {items, source:'ai', at:new Date().toISOString()}
    project.task = {...project.task, state:'done', finished:new Date().toISOString(),
                    message:(String(value.note || '').slice(0, 600) || '') + `（${cleaned.length} コマ）`}
  })
}

/** Claude Code that may only read the panel pictures in `dir`. */
async function runVision(prompt, dir, signal) {
  const file = await executable('claude')
  const args = ['-p', ...optionArgs('claude', await cliOptions('claude')), '--output-format', 'json', '--tools', 'Read', '--allowedTools', 'Read',
                '--strict-mcp-config', '--no-session-persistence']
  const stdout = await new Promise((resolve, reject) => {
    const child = spawn(file, args, {cwd:dir, stdio:['pipe', 'pipe', 'pipe'], signal:AbortSignal.any([signal, AbortSignal.timeout(20 * 60000)])})
    let out = '', err = ''
    child.stdout.on('data', c => { out += c })
    child.stderr.on('data', c => { err = (err + c).slice(-2000) })
    child.on('error', reject)
    child.on('close', code => code === 0 ? resolve(out) : reject(Error('Claude Code: ' + (err.trim() || 'コード ' + code))))
    child.stdin.end(prompt)
  })
  const data = JSON.parse(stdout)
  if (data.is_error || (data.subtype && data.subtype !== 'success')) throw Error('Claude Code: ' + String(data.result || data.subtype))
  return String(data.result || '')
}

// ---------- inspection (検品) and automatic fixes ----------

const QA_SCHEMA = `## 出力
最後に \`\`\`json で次を 1 つだけ:
{"panels":[{"id":"コマの id","ok":true|false,"problems":["日本語で具体的に（何が・どこが）"],
  "fix":{"mode":"none|regenerate|edit","prompt":"描き直すときに足す英語のプロンプト（regenerate）","instruction":"P-H3 で直す英語の指示（edit）"}}],
 "note":"ページ全体の所見を 1〜3 行（日本語）"}
- regenerate: 構図・人数・動作・取り違えなど、描き直さないと直らないとき。prompt には欠けているもの・正す点を短い英語で（例: "exactly 3 people, the small young man on the left holds a fork"）。
- edit: 絵の大半は良く、一部だけ直すとき（余計な文字や吹き出しを消す、手の破綻、表情、小道具の追加、人物を設定画の見た目にそろえる）。instruction は 1〜2 文の英語の命令文。
- 問題がなければ ok: true、mode: none。好みや細部の粗では落とさない。読者が話を追うのに困る点、人物の取り違え・欠け、主要な小道具や動作の欠け、絵に描かれた文字・吹き出し・ロゴ、目立つ体の破綻だけを問題にする。`

/** Starts the AI inspection of a page: it compares every panel picture with the name and proposes fixes. */
async function startInspect(body) {
  const id = String(body.project)
  if (tasks.has(id)) throw Error('このプロジェクトでは AI が作業中です')
  const auto = body.auto !== false
  const rounds = int(body.rounds ?? 2, '直す回数', 0, 3)
  await withProject(id, project => {
    const page = int(body.page, 'ページ', 1, project.name.pages.length)
    project.qaLoop = {page, auto, max:rounds, round:0, waiting:[]}
    project.task = {state:'running', kind:'P' + page + ' の検品', service:'claude', started:new Date().toISOString()}
  })
  runInspection(id)
  return {ok:true}
}

function runInspection(id) {
  const controller = new AbortController()
  tasks.set(id, controller)
  inspectTask(id, controller.signal)
    .catch(async error => withProject(id, project => {
      if (project.qaLoop) project.qaLoop.waiting = []
      project.task = {...project.task, state:controller.signal.aborted ? 'cancelled' : 'error',
                      message:controller.signal.aborted ? '中止しました' : error.message, finished:new Date().toISOString()}
    }).catch(() => {}))
    .finally(() => tasks.delete(id))
}

async function inspectTask(id, signal) {
  const project = await readProject(id)
  const loop = project.qaLoop
  const data = await analyze(project)
  const page = data.pages.find(p => p.number === loop.page)
  const name = project.name.pages.find(p => p.number === loop.page)
  const crops = join(PROJECTS, id, 'crops')
  await rm(crops, {recursive:true, force:true})
  await mkdir(crops, {recursive:true})
  const cast = new Map(project.name.characters.map(c => [c.id, c]))
  const blocks = []
  for (const placed of page.panels) {
    const file = project.adopted?.[placed.id]?.file
    if (!file) continue
    const panel = name.panels.find(p => p.id === placed.id)
    const picture = placed.id.replace(/[^\w.-]/g, '_') + '.png'
    await run(PYTHON, ['-I', HELPER, 'crop', join(PROJECTS, id, 'images', file), join(crops, picture), String(placed.aspect)], {timeout:60000})
    const refs = []
    for (const c of (panel.characters || []).map(c => cast.get(c.id)).filter(c => c?.ref)) {
      const ref = 'ref-' + c.id.replace(/[^\w.-]/g, '_') + extname(c.ref)
      await copyFile(join(PROJECTS, id, 'refs', c.ref), join(crops, ref)).catch(() => {})
      refs.push({name:c.name || c.id, sheet:'./' + ref})
    }
    blocks.push({id:placed.id, picture:'./' + picture, beat:panel.beat, visual:panel.visual, shot:panel.shot, angle:panel.angle,
      place:panel.place_en || panel.place, detail:panel.detail,
      characters:(panel.characters || []).map(c => ({name:cast.get(c.id)?.name || c.id, appearance:cast.get(c.id)?.appearance, pos:c.pos, pose:c.pose, expression:c.expression})),
      character_sheets:refs})
  }
  if (!blocks.length) throw Error('このページには検品する絵がありません')
  const prompt = `あなたは漫画の作画の検品係です。各コマの picture（このフォルダの画像。コマの縦横比に切り抜き済み）を Read で必ず見て、ネームの指示どおりに描けているかを確かめてください。` +
    `character_sheets があれば、その人物の設定画も Read で見て、髪型・服・顔立ちがそろっているかも確かめてください。` +
    `文字・吹き出し・効果線・頭上の漫符は後から層で載せるので、絵に無くてよい（逆に絵に文字や吹き出しが描かれていたら問題）。\n\n` +
    `<file path="references/visual-check.md">\n${await readFile(join(SKILL, 'references/visual-check.md'), 'utf8')}\n</file>\n\n` +
    `## ページ\nP${loop.page}。狙い: ${name.purpose || '—'}。検品 ${loop.round + 1} 回目。\n\n## コマ\n${JSON.stringify(blocks, null, 1)}\n\n${QA_SCHEMA}`
  const answer = await runVision(prompt, crops, signal)
  const fenced = [...answer.matchAll(/```(?:json)?\s*\n([\s\S]*?)\n```/g)].map(m => m[1])
  let value
  for (const candidate of (fenced.length ? fenced : [answer.slice(answer.indexOf('{'), answer.lastIndexOf('}') + 1)]).reverse()) {
    try { value = JSON.parse(candidate); break } catch {}
  }
  if (!Array.isArray(value?.panels)) throw Error('AI の返答から検品の結果を読み取れませんでした')
  await withProject(id, async project => {
    project.qa ??= {}
    const loop = project.qaLoop
    const queued = []
    for (const item of value.panels) {
      if (!blocks.some(b => b.id === item?.id)) continue
      const fix = item.fix && typeof item.fix === 'object' ? item.fix : {}
      const mode = ['regenerate', 'edit'].includes(fix.mode) && item.ok === false ? fix.mode : 'none'
      const result = {ok:item.ok !== false, problems:(Array.isArray(item.problems) ? item.problems : []).map(x => String(x).slice(0, 300)).slice(0, 8),
                      fix:{mode, prompt:String(fix.prompt || '').slice(0, 600), instruction:String(fix.instruction || '').slice(0, 600)},
                      file:project.adopted?.[item.id]?.file, round:loop.round + 1, at:new Date().toISOString()}
      project.qa[item.id] = result
      if (loop.auto && loop.round < loop.max && !result.ok && mode !== 'none') {
        try { queued.push((await applyFix(project, item.id, result)).id) } catch (error) { result.problems.push('自動の直しを入れられませんでした: ' + error.message) }
      }
    }
    const failed = value.panels.filter(p => p?.ok === false).length
    loop.round += 1
    loop.waiting = queued
    project.task = {...project.task, state:'done', finished:new Date().toISOString(),
      message:`検品 ${loop.round} 回目: ${blocks.length} コマ中 ${failed} コマに問題` + (queued.length ? `。${queued.length} コマを直しています（終わったら検品し直します）` : '') +
              (value.note ? '\n' + String(value.note).slice(0, 600) : '')}
  })
}

/** Queues the fix the inspection proposed: a P-H3 edit of the adopted picture, or a new Anima picture with the added prompt. */
async function applyFix(project, panelId, result) {
  if (result.fix.mode === 'edit' && result.fix.instruction) {
    const adopted = project.adopted?.[panelId]
    return editPanel(project, panelId, {take:adopted.take, file:adopted.file, instruction:result.fix.instruction, refs:true, autoAdopt:true})
  }
  const data = await analyze(project)
  return enqueuePanel(project, data, panelId, {engine:project.render.engine || 'ref', count:1, extra:result.fix.prompt, autoAdopt:true})
}

/** After the automatic fixes of a page are drawn, inspects the page again (up to the loop's limit). */
async function continueInspection(id) {
  if (tasks.has(id)) return
  let again = false
  await withProject(id, project => {
    const loop = project.qaLoop
    if (!loop?.waiting?.length) return
    const takes = Object.values(project.takes).flat().filter(t => loop.waiting.includes(t.id))
    // a fix drawn with references is followed by Anima's style pass: wait for that too
    const follows = Object.values(project.takes).flat().filter(t => t.restyleOf && loop.waiting.includes(t.restyleOf))
    if ([...takes, ...follows].some(t => ['queued', 'running'].includes(t.state))) return
    loop.waiting = []
    if (!takes.some(t => t.state === 'done')) {   // nothing new to look at
      project.task = {...project.task, message:(project.task?.message || '') + '\n自動の直しが失敗しました: ' +
        [...new Set(takes.map(t => t.message || t.state))].join(' / ')}
      return
    }
    again = true
    project.task = {state:'running', kind:'P' + loop.page + ' の検品（' + (loop.round + 1) + ' 回目）', service:'claude', started:new Date().toISOString()}
  })
  if (again) runInspection(id)
}

async function fixFromQa(body) {
  return withProject(String(body.project), async project => {
    const result = project.qa?.[body.panel]
    if (!result || result.fix.mode === 'none') throw Error('このコマに直しの提案はありません')
    const mode = body.mode === 'edit' || body.mode === 'regenerate' ? body.mode : result.fix.mode
    const take = await applyFix(project, String(body.panel), {...result, fix:{...result.fix, mode}})
    return {id:take.id}
  })
}

// ---------- character sheets ----------

// Turnaround sheet: one character from three angles (P-H3 is told to draw the person once).
const SHEET = 'character turnaround sheet, the same single character drawn three times side by side: front view, side view, back view, full body, standing straight, arms relaxed, neutral expression, plain white background'
const SHEET_NEGATIVE = 'different characters, extra people, crowd, cropped, color'
const PLATE = 'background art only, no people, wide shot, eye level, the whole place visible'
const PLATE_NEGATIVE = 'people, person, character, 1girl, 1boy, crowd, color'

async function generateSheet(body) {
  return withProject(String(body.project), async project => {
    const c = project.name.characters.find(c => c.id === body.character)
    if (!c) throw Error('人物が見つかりません')
    if (!c.appearance) throw Error('先に外見（英語）を書いてください')
    const r = project.render
    const prompt = [r.prefix, SHEET, c.appearance, project.name.meta.style].filter(Boolean).join(', ')
    const loras = [...r.loras, ...(c.lora_name ? [{name:c.lora_name, strength:c.lora_strength ?? 1}] : [])].slice(0, 4)
    const {id} = await enqueueGenerate({prompt, negative:[r.negative, NO_TEXT, SHEET_NEGATIVE].filter(Boolean).join(', '),
      width:1344, height:768, loras, count:int(body.count ?? 2, '枚数', 1, 4), steps:r.steps, cfg:r.cfg, seed:body.seed ?? null, fp8:true})
    const take = {id:randomBytes(6).toString('hex'), engine:'anima', job:id, state:'queued', files:[], created:new Date().toISOString(), prompt}
    ;(project.takes['@char:' + c.id] ??= []).unshift(take)
    pending.add(project.id)
    return {id:take.id}
  })
}

const placeId = key => createHash('sha1').update(key).digest('hex').slice(0, 10)

/** Anima draws an empty plate of a place in the work's style; the first one becomes the place's reference. */
async function enqueuePlate(project, place, count = 2) {
  const id = placeId(place.key)
  project.places ??= {}
  project.places[id] ??= {key:place.key, text:place.text}
  const r = project.render
  const prompt = [r.prefix, PLATE, place.text, project.name.meta.style].filter(Boolean).join(', ')
  const job = await enqueueGenerate({prompt, negative:[r.negative, NO_TEXT, PLATE_NEGATIVE].filter(Boolean).join(', '), width:1344, height:768,
    loras:r.loras.slice(0, 4), count, steps:r.steps, cfg:r.cfg, seed:null, fp8:true})
  const take = {id:randomBytes(6).toString('hex'), engine:'anima', job:job.id, state:'queued', files:[], created:new Date().toISOString(), prompt}
  ;(project.takes['@place:' + id] ??= []).unshift(take)
  pending.add(project.id)
  return take
}

async function generatePlate(body) {
  return withProject(String(body.project), async project => {
    const data = await analyze(project)
    const place = (data.places || []).find(p => placeId(p.key) === body.place)
    if (!place) throw Error('場所が見つかりません')
    return {id:(await enqueuePlate(project, place, int(body.count ?? 2, '枚数', 1, 4))).id}
  })
}

/** Draws every missing turnaround sheet and place plate (the groundwork for drawing with references). */
async function prepare(body) {
  const id = String(body.project)
  const project = await readProject(id)
  const data = await analyze(project)
  const waiting = key => (project.takes[key] || []).some(t => ['queued', 'running'].includes(t.state))
  const sheets = project.name.characters.filter(c => !c.ref && c.appearance && !waiting('@char:' + c.id))
  for (const c of sheets) await generateSheet({project:id, character:c.id, count:1})
  const plates = (data.places || []).filter(p => !project.places?.[placeId(p.key)]?.ref && !waiting('@place:' + placeId(p.key)))
  await withProject(id, async project => { for (const p of plates) await enqueuePlate(project, p, 1) })
  return {sheets:sheets.length, plates:plates.length}
}

async function choosePlate(body) {
  return withProject(String(body.project), async project => {
    const place = project.places?.[body.place]
    const file = basename(String(body.file || ''))
    if (!place || !existsSync(join(PROJECTS, project.id, 'images', file))) throw Error('背景が見つかりません')
    await mkdir(join(PROJECTS, project.id, 'refs'), {recursive:true})
    place.ref = 'place-' + body.place + '-' + randomBytes(3).toString('hex') + extname(file)
    await copyFile(join(PROJECTS, project.id, 'images', file), join(PROJECTS, project.id, 'refs', place.ref))
    return {ok:true}
  })
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
          if (params.has('styles')) return reply(await typesetCatalog())
          if (params.has('overlay')) {
            const id = params.get('overlay')
            const {project, analysis} = await withProject(id, async project => ({project, analysis:await analyze(project)}), false)
            return reply({panels:await overlays(project, analysis, Number(params.get('page')))})
          }
          return reply({projects:await listProjects(), services:SERVICES.map(id => ({id, label:agentNames[id] || id})),
                        loras:await listLoraNames().catch(() => [])})
        }
        if (request.headers.get('content-type')?.split(';')[0] !== 'application/json') return reply({error:'JSON が必要です'}, 415)
        const body = await request.json()
        const actions = {
          create:createProject, delete:b => deleteProject(b.project), write:startWrite, cancelWrite:b => cancelWrite(b.project),
          settings:updateSettings, character:updateCharacter, page:updatePage, panel:updatePanel,
          generate, edit, adopt, removeTake, layers:saveLayers, letter:startLetter, inspect:startInspect, fix:fixFromQa, sheet:generateSheet, plate:generatePlate, prepare, choosePlate, animaEdit,
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
    return () => { clearInterval(timer); for (const c of tasks.values()) c.abort(); worker?.kill(); worker = null }
  }, 'Manga studio sweep')
}
