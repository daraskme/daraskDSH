window.__ModuleLoader__.load({
  id:'dsh-manga-studio',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const {Button} = require('@deepseek-ai/dsh-client-ui-primitives')
    const API = '/api/darask.manga'
    const SHOTS = [['extreme_long','超ロング'],['long','ロング'],['full','フル'],['medium','ミドル'],['bust','バストアップ'],
      ['closeup','アップ'],['extreme_closeup','超アップ'],['insert','インサート']]
    const ANGLES = [['eye','アイレベル'],['high','俯瞰'],['low','煽り'],['bird','真俯瞰'],['worm','真煽り'],['dutch','斜め'],['back','後ろから'],['pov','主観']]
    const DETAILS = [['','（指定なし）'],['key','決めコマ'],['normal','通常'],['light','簡略']]
    const TEXT_TYPES = [['dialogue','台詞'],['shout','叫び'],['whisper','ささやき'],['thought','心の声'],['monologue','モノローグ'],
      ['narration','ナレーション'],['sfx','擬音'],['sign','作中の文字'],['aside','欄外']]
    const POSITIONS = [['right','右'],['center','中央'],['left','左']]
    const FORMATS = [['page','横開き（右綴じ）'],['yonkoma','4コマ'],['webtoon','縦読み']]
    const SCENES = {talk:'会話',action:'動き',gag:'ギャグ',emotion:'感情',tense:'緊張'}
    const TAKE_STATE = {queued:'待機中',running:'生成中',done:'完了',error:'エラー',cancelled:'中止'}
    const LEVEL = {FAIL:'要修正',WARN:'注意',INFO:'参考'}
    const SERIF = "'Noto Serif CJK JP','Hiragino Mincho ProN','Yu Mincho',serif"
    const SANS = "'Noto Sans CJK JP','Hiragino Sans','Yu Gothic',sans-serif"

    async function api(body) {
      const res = await fetch(API, {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})
      const data = await res.json()
      if (!res.ok) throw Error(data.error || 'エラー ' + res.status)
      return data
    }
    const get = query => fetch(API + query).then(async res => { const data = await res.json(); if (!res.ok) throw Error(data.error); return data })
    const fileUrl = (project, file, kind = 'images') => API + '?project=' + encodeURIComponent(project) + '&kind=' + kind + '&file=' + encodeURIComponent(file)
    const toBase64 = file => new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result).split(',')[1] || '')
      reader.onerror = () => reject(reader.error)
      reader.readAsDataURL(file)
    })
    const store = {
      read(key, fallback) { try { return JSON.parse(localStorage.getItem('dsh-manga.' + key)) ?? fallback } catch { return fallback } },
      write(key, value) { try { localStorage.setItem('dsh-manga.' + key, JSON.stringify(value)) } catch {} },
    }
    const pct = v => (v / 10) + '%'

    // ---------- shared state ----------

    function useProject(id) {
      const [data,setData] = React.useState(null)
      const [error,setError] = React.useState('')
      const reload = React.useCallback(async () => {
        try { setData(await get('?project=' + encodeURIComponent(id))); setError('') } catch (e) { setError(e.message) }
      }, [id])
      React.useEffect(() => {
        let stop = false, timer
        const poll = async () => { await reload(); if (!stop) timer = setTimeout(poll, 3000) }
        poll()
        return () => { stop = true; clearTimeout(timer) }
      }, [reload])
      return {data, error, reload}
    }

    function useAction(reload) {
      const [busy,setBusy] = React.useState(false)
      const [error,setError] = React.useState('')
      const [notice,setNotice] = React.useState('')
      async function act(body, done) {
        setBusy(true); setError(''); setNotice('')
        try { const result = await api(body); if (done) setNotice(done); await reload?.(); return result } catch (e) { setError(e.message) } finally { setBusy(false) }
      }
      const messages = h(React.Fragment,null,
        notice && h('p',{className:'mg-meta',role:'status'},notice),
        error && h('p',{className:'mg-error',role:'alert'},error))
      return {busy, act, messages, setError}
    }

    const adoptedFile = (project, panelId) => project.adopted?.[panelId]?.file
    const pendingTake = (project, panelId) => (project.takes?.[panelId] || []).find(t => t.state === 'queued' || t.state === 'running')

    // ---------- page rendering ----------

    const PAD = 0.15  // typeset.py: the overlay SVG reaches this far outside the panel (balloons over the frame)
    const svgUrl = svg => 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
    const padded = b => {
      const w = b[2] - b[0], hh = b[3] - b[1]
      return {left:pct(b[0] - PAD * w),top:pct(b[1] - PAD * hh),width:pct(w * (1 + 2 * PAD)),height:pct(hh * (1 + 2 * PAD))}
    }

    /** The lettering/effects overlays of one page (SVG from the server), refreshed when the project changes. */
    function useOverlays(project, number) {
      const [panels,setPanels] = React.useState({})
      React.useEffect(() => {
        let stop = false
        get('?overlay=' + encodeURIComponent(project.id) + '&page=' + number).then(d => { if (!stop) setPanels(d.panels) }).catch(() => {})
        return () => { stop = true }
      }, [project.id, number, project.updated])
      return panels
    }

    /** One page: panels at their boxes with the adopted pictures; balloons, lettering and effects as layers on top. */
    function Page({data, page, onPanel, selected, small}) {
      const {project} = data
      const name = project.name.pages.find(p => p.number === page.number)
      const beats = new Map((name?.panels || []).map(p => [p.id, p]))
      const overlays = useOverlays(project, page.number)
      return h('div',{className:'mg-page' + (small ? ' is-small' : ''),style:{aspectRatio:'1 / ' + page.aspect}},
        page.panels.map((panel, index) => {
          const file = adoptedFile(project, panel.id)
          const busy = pendingTake(project, panel.id)
          const b = panel.bbox
          const box = {left:pct(b[0]),top:pct(b[1]),width:pct(b[2] - b[0]),height:pct(b[3] - b[1])}
          return h(onPanel ? 'button' : 'div',{key:panel.id,type:onPanel ? 'button' : undefined,
              className:'mg-panel' + (selected === panel.id ? ' is-selected' : '') + (beats.get(panel.id)?.inset ? ' is-inset' : ''),
              style:box,onClick:onPanel ? () => onPanel(panel.id) : undefined,'aria-label':'コマ ' + (index + 1) + ': ' + (beats.get(panel.id)?.beat || '')},
            file ? h('img',{src:fileUrl(project.id, file),alt:'',loading:'lazy'})
              : !small && h('span',{className:'mg-beat'},(index + 1) + '. ' + (beats.get(panel.id)?.beat || '')),
            busy && h('span',{className:'mg-badge'},busy.progress || TAKE_STATE[busy.state]),
            !small && !busy && project.qa?.[panel.id] && project.qa[panel.id].file === file &&
              h('span',{className:'mg-qa' + (project.qa[panel.id].ok ? ' is-ok' : ''),title:project.qa[panel.id].problems.join('\n') || '問題なし'},project.qa[panel.id].ok ? '検品 OK' : '要確認'))
        }),
        page.panels.map(panel => overlays[panel.id] && h('img',{key:'o' + panel.id,className:'mg-overlay',alt:'',
          src:svgUrl(overlays[panel.id].svg),style:padded(panel.bbox)})))
    }

    /** Draws a page on a canvas: pictures, frames, then the overlays (all paths, so they look the same everywhere). */
    async function renderPage(data, page) {
      const W = 1654, H = Math.round(W * page.aspect)
      const canvas = document.createElement('canvas')
      canvas.width = W; canvas.height = H
      const g = canvas.getContext('2d')
      g.fillStyle = '#fff'; g.fillRect(0, 0, W, H)
      const sx = v => v / 1000 * W, sy = v => v / 1000 * H
      const load = src => new Promise(resolve => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => resolve(null); img.src = src })
      const overlays = (await get('?overlay=' + encodeURIComponent(data.project.id) + '&page=' + page.number)).panels
      for (const panel of page.panels) {
        const [x1, y1, x2, y2] = panel.bbox, x = sx(x1), y = sy(y1), w = sx(x2) - x, hh = sy(y2) - y
        const file = adoptedFile(data.project, panel.id)
        g.save(); g.beginPath(); g.rect(x, y, w, hh); g.clip()
        g.fillStyle = '#fff'; g.fillRect(x, y, w, hh)
        const img = file && await load(fileUrl(data.project.id, file))
        if (img) {
          const scale = Math.max(w / img.width, hh / img.height)
          g.drawImage(img, x + (w - img.width * scale) / 2, y + (hh - img.height * scale) / 2, img.width * scale, img.height * scale)
        }
        g.restore()
        g.lineWidth = 4; g.strokeStyle = '#000'; g.strokeRect(x, y, w, hh)
      }
      for (const panel of page.panels) {
        const o = overlays[panel.id]
        const img = o && await load(svgUrl(o.svg))
        if (!img) continue
        const [x1, y1, x2, y2] = panel.bbox, w = sx(x2) - sx(x1), hh = sy(y2) - sy(y1)
        g.drawImage(img, sx(x1) - PAD * w, sy(y1) - PAD * hh, w * (1 + 2 * PAD), hh * (1 + 2 * PAD))
      }
      return canvas
    }

    function download(name, href) {
      const link = document.createElement('a')
      link.download = name
      link.href = href
      link.click()
      if (href.startsWith('blob:')) setTimeout(() => URL.revokeObjectURL(href), 60000)
    }
    const baseName = data => (data.project.name.meta.title || 'manga').replace(/[\\/:*?"<>|]/g, '_')

    async function exportPage(data, page) {
      const canvas = await renderPage(data, page)
      download(baseName(data) + '-p' + String(page.number).padStart(3, '0') + '.png', canvas.toDataURL('image/png'))
    }

    // CBZ (zip of page images + ComicInfo.xml, read right to left), stored without compression.
    const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0 } return t })()
    const crc32 = bytes => { let c = 0xFFFFFFFF; for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0 }
    function zip(files) {
      const enc = new TextEncoder(), parts = [], central = []
      let offset = 0
      for (const file of files) {
        const name = enc.encode(file.name), data = file.data, crc = crc32(data)
        const head = new DataView(new ArrayBuffer(30))
        ;[[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [10, 0, 2], [12, 0x21, 2], [14, crc, 4], [18, data.length, 4], [22, data.length, 4], [26, name.length, 2], [28, 0, 2]]
          .forEach(([at, v, n]) => n === 4 ? head.setUint32(at, v, true) : head.setUint16(at, v, true))
        const dir = new DataView(new ArrayBuffer(46))
        ;[[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [10, 0, 2], [12, 0, 2], [14, 0x21, 2], [16, crc, 4], [20, data.length, 4], [24, data.length, 4],
          [28, name.length, 2], [30, 0, 2], [32, 0, 2], [34, 0, 2], [36, 0, 2], [38, 0, 4], [42, offset, 4]]
          .forEach(([at, v, n]) => n === 4 ? dir.setUint32(at, v, true) : dir.setUint16(at, v, true))
        parts.push(head, name, data)
        central.push(dir, name)
        offset += 30 + name.length + data.length
      }
      const size = central.reduce((n, x) => n + x.byteLength, 0)
      const end = new DataView(new ArrayBuffer(22))
      ;[[0, 0x06054b50, 4], [4, 0, 2], [6, 0, 2], [8, files.length, 2], [10, files.length, 2], [12, size, 4], [16, offset, 4], [20, 0, 2]]
        .forEach(([at, v, n]) => n === 4 ? end.setUint32(at, v, true) : end.setUint16(at, v, true))
      return new Blob([...parts, ...central, end], {type:'application/vnd.comicbook+zip'})
    }

    async function exportCbz(data, progress) {
      const files = []
      const pages = data.analysis.pages
      for (const page of pages) {
        progress?.('P' + page.number + ' / ' + pages.length + ' を組んでいます')
        const canvas = await renderPage(data, page)
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
        files.push({name:String(page.number).padStart(3, '0') + '.png', data:new Uint8Array(await blob.arrayBuffer())})
      }
      const x = v => String(v).replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]))
      const info = `<?xml version="1.0" encoding="utf-8"?>\n<ComicInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n` +
        `  <Title>${x(data.project.name.meta.title || '')}</Title>\n  <Summary>${x(data.project.brief.idea || '')}</Summary>\n` +
        `  <PageCount>${pages.length}</PageCount>\n  <LanguageISO>ja</LanguageISO>\n  <Manga>YesAndRightToLeft</Manga>\n  <BlackAndWhite>Yes</BlackAndWhite>\n</ComicInfo>\n`
      files.push({name:'ComicInfo.xml', data:new TextEncoder().encode(info)})
      download(baseName(data) + '.cbz', URL.createObjectURL(zip(files)))
      progress?.('')
    }


    // ---------- project list ----------

    function NewProject({services, open}) {
      const [form,setForm] = React.useState({title:'',idea:'',genre:'',pages:8,format:'page',rating:'general',style:'',service:'claude',write:true})
      const {busy, act, messages} = useAction()
      const set = patch => setForm(f => ({...f,...patch}))
      async function submit(e) {
        e.preventDefault()
        const created = await act({action:'create',...form})
        if (!created) return
        if (form.write) await act({action:'write',project:created.id,scope:'all'})
        open(created.id)
      }
      return h('form',{className:'mg-card',onSubmit:submit},
        h('h2',null,'新しい漫画'),
        h('label',null,h('span',null,'題名'),h('input',{value:form.title,required:true,disabled:busy,onChange:e => set({title:e.target.value})})),
        h('label',null,h('span',null,'企画・アイデア（誰が・何をして・どうなる。見せたい場面、オチ、好きなノリなど）'),
          h('textarea',{value:form.idea,rows:6,disabled:busy,placeholder:'例: 無愛想な天才パティシエの先輩と、味音痴の新人。新人の「まずい」が実は先輩の隠し味を見抜いていた――という職場ラブコメの読切。',
            onChange:e => set({idea:e.target.value})})),
        h('div',{className:'mg-grid'},
          h('label',null,h('span',null,'ジャンル'),h('input',{value:form.genre,disabled:busy,placeholder:'ラブコメ・ギャグ・ホラー…',onChange:e => set({genre:e.target.value})})),
          h('label',null,h('span',null,'ページ数'),h('input',{type:'number',min:1,max:60,value:form.pages,disabled:busy,onChange:e => set({pages:e.target.value})})),
          h('label',null,h('span',null,'形式'),h('select',{value:form.format,disabled:busy,onChange:e => set({format:e.target.value})},
            FORMATS.map(([v, l]) => h('option',{key:v,value:v},l)))),
          h('label',null,h('span',null,'対象'),h('select',{value:form.rating,disabled:busy,onChange:e => set({rating:e.target.value})},
            h('option',{value:'general'},'一般'),h('option',{value:'adult'},'成人向け（登場人物は全員成人）')))),
        h('label',null,h('span',null,'画風（英語推奨。空欄なら白黒漫画の既定）'),h('input',{value:form.style,disabled:busy,placeholder:'black-and-white Japanese manga, clean ink line art, ...',onChange:e => set({style:e.target.value})})),
        h('div',{className:'mg-row'},
          h('label',null,h('span',null,'ネームを切る AI'),h('select',{value:form.service,disabled:busy,onChange:e => set({service:e.target.value})},
            services.map(s => h('option',{key:s.id,value:s.id},s.label)))),
          h('label',{className:'mg-check'},h('input',{type:'checkbox',checked:form.write,disabled:busy,onChange:e => set({write:e.target.checked})}),
            h('span',null,'作成したらすぐ AI にネームを切らせる'))),
        h('div',{className:'mg-actions'},h(Button,{type:'submit',variant:'primary',disabled:busy || !form.title.trim()},busy ? '作成中…' : '作成')),
        h('p',{className:'mg-meta'},'ネームは ja-manga-writing の作法（コマ割りの型・演出・台詞・漫符）で切り、ネームの検査にかけます。数分かかります。'),
        messages)
    }

    function ProjectList({open}) {
      const [list,setList] = React.useState(null)
      const [error,setError] = React.useState('')
      const [creating,setCreating] = React.useState(false)
      React.useEffect(() => { get('').then(setList).catch(e => setError(e.message)) }, [])
      return h('div',{className:'mg-body'},
        error && h('p',{className:'mg-error',role:'alert'},error),
        list && h(React.Fragment,null,
          h('div',{className:'mg-toolbar'},h('strong',null,'プロジェクト'),
            h(Button,{variant:creating ? 'outline' : 'primary',size:'sm',onClick:() => setCreating(!creating)},creating ? '閉じる' : '＋ 新しい漫画')),
          (creating || !list.projects.length) && h(NewProject,{services:list.services,open}),
          h('div',{className:'mg-projects'},list.projects.map(p => h('button',{key:p.id,type:'button',className:'mg-project',onClick:() => open(p.id)},
            h('div',{className:'mg-cover'},p.cover ? h('img',{src:fileUrl(p.id, p.cover),alt:''}) : h('span',null,p.title.slice(0, 1))),
            h('strong',null,p.title),
            h('small',null,p.pages + ' ページ' + (p.task ? ' · ' + p.task.kind + '中' : '') + ' · ' + new Date(p.updated).toLocaleString()),
            p.idea && h('small',{className:'mg-clamp'},p.idea))))))
    }

    // ---------- project ----------

    function TaskStatus({project, running, act, busy}) {
      const t = project.task
      if (!t) return null
      return h('div',{className:'mg-task mg-' + t.state,role:'status'},
        h('strong',null,t.kind + ' · ' + ({running:'作業中…',done:'完了',error:'エラー',cancelled:'中止'}[t.state] || t.state)),
        h('span',{className:'mg-meta'},' ' + (t.service || '') + (t.state === 'running' ? ' · ' + Math.round((Date.now() - Date.parse(t.started)) / 60000) + ' 分経過' : '')),
        t.message && h('p',{className:t.state === 'error' ? 'mg-error' : 'mg-meta'},t.message),
        running && h(Button,{variant:'outline',size:'sm',disabled:busy,onClick:() => act({action:'cancelWrite',project:project.id})},'中止'))
    }

    function WriteBox({data, act, busy, page}) {
      const {project} = data
      const [instruction,setInstruction] = React.useState('')
      const [service,setService] = React.useState(project.brief.service)
      const [services,setServices] = React.useState([])
      React.useEffect(() => { get('').then(d => setServices(d.services)).catch(() => {}) }, [])
      const running = project.task?.state === 'running'
      const first = !project.name.pages.length
      return h('div',{className:'mg-card'},
        h('h3',null,page ? 'AI でこのページを直す' : first ? 'AI でネームを切る' : 'AI でネームを書き直す'),
        h('textarea',{value:instruction,rows:3,disabled:busy || running,
          placeholder:page ? '例: 2 コマ目の台詞を削って表情で見せる。最後のコマでめくりの引きを強く' : first ? '追加の指示（空欄可）' : '例: 中盤がだれるので 4〜5 ページを詰め、オチの前に一度ずらしを入れる',
          onChange:e => setInstruction(e.target.value)}),
        h('div',{className:'mg-row'},
          h('select',{value:service,'aria-label':'AI',disabled:busy || running,onChange:e => setService(e.target.value)},
            (services.length ? services : [{id:service,label:service}]).map(s => h('option',{key:s.id,value:s.id},s.label))),
          h(Button,{variant:'primary',size:'sm',disabled:busy || running || (page && !instruction.trim()),
            onClick:() => act({action:'write',project:project.id,scope:page ? 'page' : 'all',page,instruction,service},'AI に頼みました。終わるまで数分かかります')},
            running ? 'AI が作業中' : page ? 'このページを直す' : first ? 'ネームを切る' : '書き直す')),
        !page && !first && h('p',{className:'mg-meta'},'書き直すと、コマの id が変わったコマの絵は外れます（絵の履歴は残ります）。'))
    }

    function Issues({issues, where}) {
      const list = (issues || []).filter(i => !where || i.where.startsWith(where))
      if (!list.length) return h('p',{className:'mg-meta'},'検査で指摘はありません。')
      return h('ul',{className:'mg-issues'},list.map((i, k) => h('li',{key:k,className:'mg-' + i.level},
        h('b',null,(LEVEL[i.level] || i.level) + ' ' + i.code),' ',h('span',{className:'mg-meta'},i.where),' ',i.message)))
    }

    function Settings({data, act, busy}) {
      const {project} = data
      const [form,setForm] = React.useState(() => ({title:project.name.meta.title,style:project.name.meta.style,idea:project.brief.idea,
        genre:project.brief.genre,pages:project.brief.pages,prefix:project.render.prefix,negative:project.render.negative,
        steps:project.render.steps,cfg:project.render.cfg,loras:project.render.loras}))
      const [loras,setLoras] = React.useState([])
      React.useEffect(() => { get('').then(d => setLoras(d.loras)).catch(() => {}) }, [])
      const set = patch => setForm(f => ({...f,...patch}))
      const setLora = (i, patch) => set({loras:form.loras.map((l, k) => k === i ? {...l,...patch} : l)})
      return h('form',{className:'mg-card',onSubmit:e => { e.preventDefault(); act({action:'settings',project:project.id,...form},'保存しました') }},
        h('h3',null,'作品の設定'),
        h('label',null,h('span',null,'題名'),h('input',{value:form.title,onChange:e => set({title:e.target.value})})),
        h('label',null,h('span',null,'企画・アイデア'),h('textarea',{rows:4,value:form.idea,onChange:e => set({idea:e.target.value})})),
        h('div',{className:'mg-grid'},
          h('label',null,h('span',null,'ジャンル'),h('input',{value:form.genre,onChange:e => set({genre:e.target.value})})),
          h('label',null,h('span',null,'ページ数（次に切るとき）'),h('input',{type:'number',min:1,max:60,value:form.pages,onChange:e => set({pages:e.target.value})}))),
        h('label',null,h('span',null,'画風ブロック（全コマ共通・英語）'),h('textarea',{rows:3,value:form.style,onChange:e => set({style:e.target.value})})),
        h('label',null,h('span',null,'Anima のプロンプトの先頭'),h('input',{value:form.prefix,onChange:e => set({prefix:e.target.value})})),
        h('label',null,h('span',null,'ネガティブプロンプト（Anima）'),h('input',{value:form.negative,onChange:e => set({negative:e.target.value})})),
        h('fieldset',null,h('legend',null,'全コマにかける Anima LoRA（画風など）'),
          form.loras.map((l, i) => h('div',{key:i,className:'mg-row'},
            h('select',{value:l.name,'aria-label':'LoRA',onChange:e => setLora(i,{name:e.target.value})},h('option',{value:''},'選択'),
              loras.map(n => h('option',{key:n,value:n},n.replace(/\.safetensors$/,'')))),
            h('input',{type:'number',step:0.05,min:-2,max:2,value:l.strength,className:'mg-narrow','aria-label':'強さ',onChange:e => setLora(i,{strength:e.target.value})}),
            h(Button,{type:'button',size:'sm',variant:'outline',onClick:() => set({loras:form.loras.filter((_, k) => k !== i)})},'外す'))),
          form.loras.length < 4 && h(Button,{type:'button',size:'sm',variant:'outline',className:'mg-start',onClick:() => set({loras:[...form.loras,{name:'',strength:1}]})},'LoRA を追加')),
        h('div',{className:'mg-grid'},
          h('label',null,h('span',null,'step'),h('input',{type:'number',min:1,max:100,value:form.steps,onChange:e => set({steps:e.target.value})})),
          h('label',null,h('span',null,'CFG'),h('input',{type:'number',min:1,max:20,step:0.1,value:form.cfg,onChange:e => set({cfg:e.target.value})}))),
        h('div',{className:'mg-actions'},h(Button,{type:'submit',variant:'primary',size:'sm',disabled:busy},'保存')))
    }

    function CharacterForm({data, act, busy, character, done}) {
      const {project} = data
      const [form,setForm] = React.useState(() => ({id:'',name:'',appearance:'',voice:'',lora_name:'',lora_strength:1,...character}))
      const [loras,setLoras] = React.useState([])
      const [refFile,setRefFile] = React.useState(null)
      React.useEffect(() => { get('').then(d => setLoras(d.loras)).catch(() => {}) }, [])
      const set = patch => setForm(f => ({...f,...patch}))
      async function submit(e) {
        e.preventDefault()
        const c = {id:form.id,name:form.name,appearance:form.appearance,voice:form.voice,lora_name:form.lora_name,lora_strength:form.lora_strength}
        if (refFile) Object.assign(c, {ref_name:refFile.name,ref_data:await toBase64(refFile)})
        const ok = await act({action:'character',project:project.id,original:character?.id,character:c},'保存しました')
        if (ok) done?.()
      }
      return h('form',{className:'mg-character',onSubmit:submit},
        h('div',{className:'mg-ref'},character?.ref ? h('img',{src:fileUrl(project.id, character.ref, 'refs'),alt:'設定画'}) : h('span',null,'設定画なし'),
          h('label',{className:'mg-file'},h('input',{type:'file',accept:'image/png,image/jpeg,image/webp',onChange:e => setRefFile(e.target.files[0] || null)}),
            h('span',null,refFile ? refFile.name : '設定画を選ぶ'))),
        h('div',{className:'mg-fields'},
          h('div',{className:'mg-grid'},
            h('label',null,h('span',null,'id（英字）'),h('input',{value:form.id,required:true,onChange:e => set({id:e.target.value})})),
            h('label',null,h('span',null,'名前'),h('input',{value:form.name,onChange:e => set({name:e.target.value})}))),
          h('label',null,h('span',null,'外見（英語。全コマで同じ文面を使う）'),h('textarea',{rows:2,value:form.appearance,onChange:e => set({appearance:e.target.value})})),
          h('label',null,h('span',null,'口調'),h('input',{value:form.voice,onChange:e => set({voice:e.target.value})})),
          h('div',{className:'mg-row'},
            h('select',{value:form.lora_name || '','aria-label':'キャラ LoRA',onChange:e => set({lora_name:e.target.value})},h('option',{value:''},'キャラ LoRA なし'),
              loras.map(n => h('option',{key:n,value:n},n.replace(/\.safetensors$/,'')))),
            h('input',{type:'number',step:0.05,min:-2,max:2,value:form.lora_strength ?? 1,className:'mg-narrow','aria-label':'強さ',onChange:e => set({lora_strength:e.target.value})})),
          character && h(Sheets,{data,act,busy,character}),
          h('div',{className:'mg-actions'},h(Button,{type:'submit',size:'sm',variant:'primary',disabled:busy},'保存'),
            character && h(Button,{type:'button',size:'sm',variant:'outline',disabled:busy,onClick:() => { if (confirm(form.name + ' を人物表から外しますか？')) act({action:'character',project:project.id,remove:character.id}) }},'外す'))))
    }

    /** Character sheets drawn by Anima from the appearance; the first becomes the reference when there is none. */
    function Sheets({data, act, busy, character}) {
      const {project} = data
      const takes = project.takes?.['@char:' + character.id] || []
      const files = takes.flatMap(t => t.files.map(file => ({file, take:t.id})))
      const waiting = takes.find(t => t.state === 'queued' || t.state === 'running')
      return h('div',{className:'mg-sheets'},
        h('div',{className:'mg-actions'},
          h(Button,{type:'button',size:'sm',variant:'outline',disabled:busy || !character.appearance || !!waiting,
            onClick:() => act({action:'sheet',project:project.id,character:character.id,count:2},'設定画を Anima のキューに入れました')},
            waiting ? '設定画を描いています（' + (waiting.progress || TAKE_STATE[waiting.state]) + '）' : 'AI で設定画を描く（2 枚）'),
          h('span',{className:'mg-meta'},'外見の英文から全身の立ち絵を描き、設定画がまだなければ 1 枚目を使います。')),
        files.length > 0 && h('div',{className:'mg-sheet-list'},files.map(({file, take}) => h('figure',{key:file},
          h('a',{href:fileUrl(project.id, file),target:'_blank',rel:'noopener'},h('img',{src:fileUrl(project.id, file),alt:'設定画の候補',loading:'lazy'})),
          h('figcaption',null,
            h('button',{type:'button',disabled:busy,onClick:() => act({action:'character',project:project.id,original:character.id,character:{...character,ref_from:file}},'設定画にしました')},'設定画にする'),
            h('button',{type:'button',disabled:busy,onClick:() => act({action:'removeTake',project:project.id,panel:'@char:' + character.id,take,file})},'消す'))))))
    }

    function Characters({data, act, busy}) {
      const [adding,setAdding] = React.useState(false)
      return h('div',{className:'mg-card'},
        h('h3',null,'人物'),
        h('p',{className:'mg-meta'},'設定画を付けると、P-H3 で参照生成・修正するときにその人物の見た目を渡します。キャラ LoRA はその人物が出るコマの Anima 生成にかけます。'),
        data.project.name.characters.map(c => h(CharacterForm,{key:c.id + (c.ref || ''),data,act,busy,character:c})),
        adding ? h(CharacterForm,{data,act,busy,done:() => setAdding(false)})
          : h(Button,{size:'sm',variant:'outline',className:'mg-start',onClick:() => setAdding(true)},'＋ 人物を追加'))
    }

    function CbzButton({data}) {
      const [status,setStatus] = React.useState('')
      return h(React.Fragment,null,
        h(Button,{size:'sm',variant:'outline',disabled:!!status,onClick:() => exportCbz(data, setStatus).catch(e => setStatus('失敗: ' + e.message))},status || 'CBZ で書き出す'))
    }

    function ProjectView({data, act, busy, messages, go}) {
      const {project, analysis, running} = data
      const [tab,setTab] = React.useState('pages')
      const missing = analysis.pages.flatMap(p => p.panels).filter(p => !adoptedFile(project, p.id) && !pendingTake(project, p.id)).map(p => p.id)
      const fails = (analysis.issues || []).filter(i => i.level === 'FAIL').length
      return h('div',{className:'mg-body'},
        h(TaskStatus,{project,running,act,busy}),
        messages,
        h('div',{className:'mg-tabs',role:'tablist'},[['pages','ページ'],['plan','企画メモ'],['cast','人物'],['check','検査' + (fails ? '（' + fails + '）' : '')],['settings','設定']]
          .map(([key, label]) => h('button',{key,type:'button',role:'tab','aria-selected':tab === key,className:tab === key ? 'is-active' : '',onClick:() => setTab(key)},label))),
        tab === 'pages' && h(React.Fragment,null,
          analysis.error && h('p',{className:'mg-error'},analysis.error),
          !project.name.pages.length && h(WriteBox,{data,act,busy}),
          analysis.pages.length > 0 && h('div',{className:'mg-toolbar'},
            h('span',{className:'mg-meta'},analysis.pages.length + ' ページ · ' + analysis.pages.reduce((n, p) => n + p.panels.length, 0) + ' コマ · 絵のないコマ ' + missing.length),
            h(Button,{size:'sm',variant:'primary',disabled:busy || !missing.length,
              onClick:() => act({action:'generate',project:project.id,panels:missing,engine:'anima',count:2},missing.length + ' コマを Anima のキューに入れました')},'絵のないコマを全部描く'),
            h(CbzButton,{data})),
          h('div',{className:'mg-spreads'},spreads(analysis.pages).map((pair, i) => h('div',{key:i,className:'mg-spread'},
            pair.map(page => page ? h('button',{key:page.number,type:'button',className:'mg-thumb',onClick:() => go({page:page.number})},
              h(Page,{data,page,small:true}),h('span',null,'P' + page.number)) : h('span',{key:'blank' + i,className:'mg-thumb is-blank'}))))),
          project.name.pages.length > 0 && h(WriteBox,{data,act,busy})),
        tab === 'plan' && h('div',{className:'mg-card'},h('h3',null,'企画メモ（AI がネームを切る前に書いた設計）'),
          h('pre',{className:'mg-plan'},project.plan || 'まだありません。'),
          h('h3',null,'ページごとの狙い'),
          h('ol',{className:'mg-purposes'},project.name.pages.map(p => h('li',{key:p.number},h('b',null,p.purpose || '（狙いなし）'),p.turn && h('div',{className:'mg-meta'},'めくり: ' + p.turn))))),
        tab === 'cast' && h(Characters,{data,act,busy}),
        tab === 'check' && h('div',{className:'mg-card'},h('h3',null,'ネームの検査（ja-manga-writing の name_lint）'),
          h('p',{className:'mg-meta'},'点数は出しません。読み順・めくり・同じショットの連続・台詞の量・話者・漫符などの機械検査です。'),
          h(Issues,{issues:analysis.issues})),
        tab === 'settings' && h(React.Fragment,null,h(Settings,{data,act,busy}),
          h('div',{className:'mg-card'},h(Button,{variant:'outline',size:'sm',disabled:busy,onClick:async () => {
            if (confirm('「' + project.name.meta.title + '」を削除しますか？（.runtime/manga/trash に移します）')) { await act({action:'delete',project:project.id}); go(null) }
          }},'このプロジェクトを削除'))))
    }

    /** Right-bound spreads: pages on the left side open the spread with a blank on its right when needed. */
    function spreads(pages) {
      const out = []
      let pair = []
      for (const page of pages) {
        if (page.side === 'right' && pair.length) { out.push(pair); pair = [] }
        if (page.side === 'left' && !pair.length) pair.push(null)
        pair.push(page)
        if (pair.length === 2) { out.push(pair.reverse()); pair = [] }
      }
      if (pair.length) out.push(pair.length === 1 ? [null, pair[0]] : pair.reverse())
      return out
    }

    // ---------- page ----------

    function LayoutPicker({count, current, choose}) {
      const [list,setList] = React.useState(null)
      const [open,setOpen] = React.useState(false)
      React.useEffect(() => { if (open && !list) get('?layouts=1').then(d => setList(d.layouts)).catch(() => setList([])) }, [open])
      const fit = (list || []).filter(x => x.boxes.length === count)
      return h('div',{className:'mg-layouts'},
        h(Button,{size:'sm',variant:'outline',onClick:() => setOpen(!open)},open ? 'コマ割りの型を閉じる' : 'コマ割りの型を選ぶ（' + count + ' コマ）'),
        open && h('div',{className:'mg-layout-list'},
          !list ? h('p',{className:'mg-meta'},'読み込み中…') : !fit.length ? h('p',{className:'mg-meta'},count + ' コマの型はありません。コマを足すか減らしてください。') :
          fit.map(x => h('button',{key:x.id,type:'button',className:'mg-layout' + (current === x.id ? ' is-selected' : ''),onClick:() => choose(x.id),
              title:(x.purpose || '') + (x.avoid ? '\n避ける: ' + x.avoid : '') + (x.when ? '\n' + x.when.join('\n') : '')},
            h('svg',{viewBox:'0 0 100 141','aria-hidden':true},x.boxes.map((b, i) => h('rect',{key:i,x:7 + b[0] * 0.086,y:6 + b[1] * 0.123,
              width:(b[2] - b[0]) * 0.086,height:(b[3] - b[1]) * 0.123,fill:'#fff',stroke:'currentColor',strokeWidth:1.2}))),
            h('strong',null,x.id),h('small',null,x.name),
            h('small',{className:'mg-meta'},(x.scenes || []).map(s => SCENES[s] || s).join('・') + (x.evidence?.pages ? ' · 実例 ' + x.evidence.pages + 'p' : ''))))),
        current && h(Button,{size:'sm',variant:'outline',onClick:() => choose('')},'型を外す（段組みに戻す）'))
    }

    function PageView({data, number, act, busy, messages, go}) {
      const {project, analysis} = data
      const page = analysis.pages.find(p => p.number === number)
      const name = project.name.pages.find(p => p.number === number)
      const [purpose,setPurpose] = React.useState(name?.purpose || '')
      React.useEffect(() => setPurpose(name?.purpose || ''), [number])
      if (!page || !name) return h('div',{className:'mg-body'},h('p',{className:'mg-meta'},'ページが見つかりません。'),h(Button,{size:'sm',variant:'outline',onClick:() => go({page:null})},'戻る'))
      const last = analysis.pages[analysis.pages.length - 1].number
      const missing = page.panels.filter(p => !adoptedFile(project, p.id) && !pendingTake(project, p.id)).map(p => p.id)
      return h('div',{className:'mg-body mg-pagebody'},
        h('div',{className:'mg-pagecol'},
          h('div',{className:'mg-toolbar'},
            h(Button,{size:'sm',variant:'outline',disabled:number >= last,onClick:() => go({page:number + 1})},'← 次'),
            h('strong',null,'P' + number + '（' + (page.side === 'left' ? '左' : '右') + 'ページ）'),
            h(Button,{size:'sm',variant:'outline',disabled:number <= 1,onClick:() => go({page:number - 1})},'前 →')),
          h(Page,{data,page,onPanel:id => go({page:number,panel:id})}),
          h('p',{className:'mg-meta'},'コマを押すと、そのコマの絵を作る・直す画面に移ります。右綴じなので「次」は左へ。')),
        h('div',{className:'mg-side'},
          messages,
          h('div',{className:'mg-card'},
            h('label',null,h('span',null,'このページの狙い'),h('textarea',{rows:2,value:purpose,onChange:e => setPurpose(e.target.value),
              onBlur:() => purpose !== (name.purpose || '') && act({action:'page',project:project.id,page:number,purpose})})),
            name.turn && h('p',{className:'mg-meta'},'めくり: ' + name.turn),
            h('div',{className:'mg-actions'},
              h(Button,{size:'sm',variant:'primary',disabled:busy || !missing.length,onClick:() => act({action:'generate',project:project.id,panels:missing,engine:'anima',count:2},'キューに入れました')},
                '絵のないコマを描く（' + missing.length + '）'),
              h(Button,{size:'sm',variant:'outline',onClick:() => exportPage(data, page)},'PNG で保存'))),
          h('div',{className:'mg-card'},
            h('h3',null,'コマ割り'),
            h(LayoutPicker,{count:name.panels.length,current:name.layout,choose:layout => act({action:'page',project:project.id,page:number,layout})}),
            h('ol',{className:'mg-panel-list'},name.panels.map(p => h('li',{key:p.id},
              h('button',{type:'button',className:'mg-link',onClick:() => go({page:number,panel:p.id})},p.beat || p.id),
              h('small',{className:'mg-meta'},' ' + (SHOTS.find(s => s[0] === p.shot)?.[1] || p.shot || '') + (p.detail === 'key' ? ' · 決め' : '')),
              h('button',{type:'button',className:'mg-x','aria-label':'このコマを消す',disabled:busy,
                onClick:() => confirm('このコマを消しますか？') && act({action:'page',project:project.id,page:number,removePanel:p.id})},'×')))),
            h(Button,{size:'sm',variant:'outline',className:'mg-start',disabled:busy,onClick:() => act({action:'page',project:project.id,page:number,addPanel:true})},'＋ コマを足す'),
            name.tiers && !name.layout && h('p',{className:'mg-meta'},'段の高さ比: ' + name.tiers.join(' / '))),
          h('div',{className:'mg-card'},h('h3',null,'検査'),h(Issues,{issues:analysis.issues,where:'P' + number})),
          h(InspectBox,{data,act,busy,page:number}),
          h(LetterBox,{data,act,busy,page:number}),
          h(WriteBox,{data,act,busy,page:number})))
    }

    // ---------- panel ----------

    function PanelEditor({data, panel, act, busy}) {
      const {project} = data
      const [form,setForm] = React.useState(() => structuredClone(panel))
      React.useEffect(() => setForm(structuredClone(panel)), [panel.id, JSON.stringify(panel)])
      const set = patch => setForm(f => ({...f,...patch}))
      const cast = project.name.characters
      const chars = form.characters || [], texts = form.text || []
      const setChar = (i, patch) => set({characters:chars.map((c, k) => k === i ? {...c,...patch} : c)})
      const setText = (i, patch) => set({text:texts.map((t, k) => k === i ? {...t,...patch} : t)})
      const save = () => act({action:'panel',project:project.id,panel:panel.id,patch:{
        beat:form.beat,visual:form.visual,shot:form.shot,angle:form.angle,detail:form.detail || '',place:form.place,place_en:form.place_en,
        extra_prompt:form.extra_prompt || '',characters:chars,text:texts}},'保存しました')
      return h('form',{className:'mg-card',onSubmit:e => { e.preventDefault(); save() }},
        h('h3',null,'コマの中身'),
        h('label',null,h('span',null,'このコマで起きること（beat）'),h('textarea',{rows:2,value:form.beat || '',onChange:e => set({beat:e.target.value})})),
        h('label',null,h('span',null,'絵（visual・英語）'),h('textarea',{rows:3,value:form.visual || '',onChange:e => set({visual:e.target.value})})),
        h('div',{className:'mg-grid'},
          h('label',null,h('span',null,'ショット'),h('select',{value:form.shot || '',onChange:e => set({shot:e.target.value})},h('option',{value:''},'—'),SHOTS.map(([v, l]) => h('option',{key:v,value:v},l)))),
          h('label',null,h('span',null,'アングル'),h('select',{value:form.angle || '',onChange:e => set({angle:e.target.value})},h('option',{value:''},'—'),ANGLES.map(([v, l]) => h('option',{key:v,value:v},l)))),
          h('label',null,h('span',null,'描き込み'),h('select',{value:form.detail || '',onChange:e => set({detail:e.target.value})},DETAILS.map(([v, l]) => h('option',{key:v,value:v},l)))),
          h('label',null,h('span',null,'場所（英語）'),h('input',{value:form.place_en || '',placeholder:form.place || '',onChange:e => set({place_en:e.target.value})}))),
        h('fieldset',null,h('legend',null,'人物'),
          chars.map((c, i) => h('div',{key:i,className:'mg-subrow'},
            h('select',{value:c.id || '','aria-label':'人物',onChange:e => setChar(i,{id:e.target.value})},h('option',{value:''},'—'),cast.map(x => h('option',{key:x.id,value:x.id},x.name || x.id))),
            h('select',{value:c.pos || '','aria-label':'位置',onChange:e => setChar(i,{pos:e.target.value})},h('option',{value:''},'位置'),POSITIONS.map(([v, l]) => h('option',{key:v,value:v},l))),
            h('input',{value:c.pose || '',placeholder:'pose（英語）',onChange:e => setChar(i,{pose:e.target.value})}),
            h('input',{value:c.expression || '',placeholder:'expression（英語）',onChange:e => setChar(i,{expression:e.target.value})}),
            h('input',{value:(c.manpu || []).join(', '),placeholder:'漫符（例: sweat_drop）',onChange:e => setChar(i,{manpu:e.target.value.split(/[,、\s]+/).filter(Boolean)})}),
            h('button',{type:'button',className:'mg-x','aria-label':'外す',onClick:() => set({characters:chars.filter((_, k) => k !== i)})},'×'))),
          h(Button,{type:'button',size:'sm',variant:'outline',className:'mg-start',onClick:() => set({characters:[...chars,{id:cast[0]?.id || '',pos:'center'}]})},'＋ 人物')),
        h('fieldset',null,h('legend',null,'台詞・文字（改行は ／）'),
          texts.map((t, i) => h('div',{key:i,className:'mg-subrow'},
            h('select',{value:t.type || 'dialogue','aria-label':'種類',onChange:e => setText(i,{type:e.target.value})},TEXT_TYPES.map(([v, l]) => h('option',{key:v,value:v},l))),
            h('select',{value:t.speaker || '','aria-label':'話者',onChange:e => setText(i,{speaker:e.target.value || undefined})},h('option',{value:''},'話者なし'),cast.map(x => h('option',{key:x.id,value:x.id},x.name || x.id))),
            h('input',{className:'mg-wide',value:Array.isArray(t.lines) ? t.lines.join('／') : t.text || '',onChange:e => setText(i,{text:e.target.value,lines:undefined})}),
            h('button',{type:'button',className:'mg-x','aria-label':'消す',onClick:() => set({text:texts.filter((_, k) => k !== i)})},'×'))),
          h(Button,{type:'button',size:'sm',variant:'outline',className:'mg-start',onClick:() => set({text:[...texts,{type:'dialogue',text:''}]})},'＋ 台詞')),
        h('label',null,h('span',null,'このコマだけの追加プロンプト（英語・任意）'),h('input',{value:form.extra_prompt || '',onChange:e => set({extra_prompt:e.target.value})})),
        h('div',{className:'mg-actions'},h(Button,{type:'submit',size:'sm',variant:'primary',disabled:busy},'保存')))
    }

    function Takes({data, panelId, act, busy}) {
      const {project} = data
      const takes = project.takes?.[panelId] || []
      const adopted = project.adopted?.[panelId]
      const [fixing,setFixing] = React.useState(null)  // {take, file}
      const [instruction,setInstruction] = React.useState('')
      const [refs,setRefs] = React.useState(true)
      const {panel} = (() => { for (const page of project.name.pages) for (const p of page.panels) if (p.id === panelId) return {panel:p}; return {} })()
      const cast = project.name.characters.filter(c => (panel?.characters || []).some(x => x.id === c.id))
      if (!takes.length) return h('p',{className:'mg-meta'},'まだ絵がありません。')
      return h('div',{className:'mg-takes'},
        takes.map(take => h('div',{key:take.id,className:'mg-take'},
          h('div',{className:'mg-meta'},({anima:'Anima',ph3:'P-H3 参照','ph3-edit':'P-H3 修正'}[take.engine] || take.engine) + ' · ' +
            (take.state === 'done' ? new Date(take.created).toLocaleString() : take.progress || TAKE_STATE[take.state]) + (take.instruction ? ' · 「' + take.instruction + '」' : '')),
          take.message && h('p',{className:'mg-error'},take.message),
          h('div',{className:'mg-files'},take.files.map(file => h('figure',{key:file,className:adopted?.file === file ? 'is-adopted' : ''},
            h('a',{href:fileUrl(project.id, file),target:'_blank',rel:'noopener'},h('img',{src:fileUrl(project.id, file),alt:'候補',loading:'lazy'})),
            h('figcaption',null,
              adopted?.file === file ? h('b',null,'採用中') : h('button',{type:'button',disabled:busy,onClick:() => act({action:'adopt',project:project.id,panel:panelId,take:take.id,file})},'採用'),
              h('button',{type:'button',disabled:busy,onClick:() => { setFixing({take:take.id,file}); setInstruction('') }},'P-H3 で直す'),
              cast.length > 0 && h('select',{value:'','aria-label':'設定画にする',disabled:busy,onChange:e => e.target.value &&
                  act({action:'character',project:project.id,original:e.target.value,character:{...cast.find(c => c.id === e.target.value),ref_from:file}},'設定画にしました')},
                h('option',{value:''},'設定画にする…'),cast.map(c => h('option',{key:c.id,value:c.id},c.name || c.id))),
              h('button',{type:'button',disabled:busy,onClick:() => act({action:'removeTake',project:project.id,panel:panelId,take:take.id,file})},'消す')),
            fixing?.file === file && h('div',{className:'mg-fix'},
              h('textarea',{rows:2,value:instruction,autoFocus:true,placeholder:'直し方（英語が確実）例: Make her smile slightly. Remove the extra hand. Add speed lines behind him.',
                onChange:e => setInstruction(e.target.value)}),
              h('label',{className:'mg-check'},h('input',{type:'checkbox',checked:refs,onChange:e => setRefs(e.target.checked)}),h('span',null,'人物の設定画も渡す')),
              cast.some(c => c.ref) && h('button',{type:'button',className:'mg-link',onClick:() => { setRefs(true); setInstruction('Make ' + cast.filter(c => c.ref).map(c => c.name || c.id).join(' and ') +
                ' look exactly like the character sheets (face, hairstyle, outfit). Keep the pose, composition, background and everything else unchanged.') }},'人物を設定画にそろえる（指示を入れる）'),
              h('div',{className:'mg-actions'},
                h(Button,{size:'sm',variant:'primary',disabled:busy || !instruction.trim(),onClick:async () => {
                  if (await act({action:'edit',project:project.id,panel:panelId,take:take.id,file,instruction,refs},'P-H3 の修正をキューに入れました')) setFixing(null)
                }},'修正する'),
                h(Button,{size:'sm',variant:'outline',onClick:() => setFixing(null)},'やめる')))))),
          (take.state === 'queued' || take.state === 'running' || !take.files.length) &&
            h('button',{type:'button',className:'mg-link',disabled:busy,onClick:() => act({action:'removeTake',project:project.id,panel:panelId,take:take.id})},
              take.state === 'queued' || take.state === 'running' ? '取り消す' : '消す'))))
    }

    // ---------- lettering and effects layers ----------

    const LAYER_LABEL = {balloon:'吹き出し',sfx:'書き文字',text:'文字',effect:'効果',mark:'漫符'}
    const BALLOON_STYLES = [['speech','ふつう'],['shout','叫び（トゲ）'],['whisper','ささやき（点線）'],['thought','心の声（雲）'],['narration','ナレーション（四角）'],['monologue','モノローグ（枠なし）']]
    let catalogPromise
    function useCatalog() {
      const [catalog,setCatalog] = React.useState(null)
      React.useEffect(() => { (catalogPromise ??= get('?styles=1')).then(setCatalog).catch(() => { catalogPromise = undefined }) }, [])
      return catalog
    }

    function InspectBox({data, act, busy, page}) {
      const {project, analysis} = data
      const [auto,setAuto] = React.useState(true)
      const running = project.task?.state === 'running'
      const placed = analysis.pages.find(p => p.number === page)?.panels || []
      const results = placed.map(p => [p.id, project.qa?.[p.id]]).filter(([, r]) => r)
      const loop = project.qaLoop?.page === page ? project.qaLoop : null
      return h('div',{className:'mg-card'},
        h('h3',null,'AI で検品'),
        h('p',{className:'mg-meta'},'Claude が各コマの絵をネーム（人物・人数・動作・小道具・ショット）と設定画に照らして見て、問題のあるコマに直し方を出します。'),
        h('label',{className:'mg-check'},h('input',{type:'checkbox',checked:auto,disabled:busy || running,onChange:e => setAuto(e.target.checked)}),
          h('span',null,'問題のあるコマは自動で直し、仕上がったら検品し直す（最大 2 回）')),
        h('div',{className:'mg-actions'},h(Button,{size:'sm',variant:'primary',disabled:busy || running || !placed.some(p => adoptedFile(project, p.id)),
          onClick:() => act({action:'inspect',project:project.id,page,auto,rounds:2},'検品を頼みました')},running ? 'AI が作業中' : '検品する'),
          loop?.waiting?.length > 0 && h('span',{className:'mg-meta'},loop.waiting.length + ' コマを直しています（' + loop.round + ' / ' + (loop.max + 1) + ' 回目の後）')),
        results.length > 0 && h('ul',{className:'mg-qa-list'},results.map(([id, r]) => h('li',{key:id,className:r.ok ? 'is-ok' : ''},
          h('b',null,id + ' ' + (r.ok ? '問題なし' : '要確認')),
          r.problems.length > 0 && h('span',{className:'mg-meta'},' ' + r.problems.join(' / ')),
          r.file !== adoptedFile(project, id) && h('span',{className:'mg-meta'},'（この後、絵が替わっています）')))))
    }

    function QaResult({data, panelId, act, busy}) {
      const {project} = data
      const r = project.qa?.[panelId]
      if (!r) return null
      const current = r.file === adoptedFile(project, panelId)
      return h('div',{className:'mg-card'},
        h('h3',null,'検品の結果（' + r.round + ' 回目）' + (current ? '' : ' · 絵が替わったので古い結果')),
        r.ok ? h('p',{className:'mg-meta'},'問題なし') : h('ul',{className:'mg-qa-list'},r.problems.map((x, i) => h('li',{key:i},x))),
        !r.ok && r.fix.mode !== 'none' && h(React.Fragment,null,
          h('p',{className:'mg-prompt'},(r.fix.mode === 'edit' ? 'P-H3 で直す: ' + r.fix.instruction : '描き直す（足すプロンプト）: ' + r.fix.prompt)),
          h('div',{className:'mg-actions'},
            h(Button,{size:'sm',variant:'primary',disabled:busy || !current,onClick:() => act({action:'fix',project:project.id,panel:panelId},'直しをキューに入れました')},'提案どおり直す'),
            r.fix.mode === 'edit' && r.fix.prompt && h(Button,{size:'sm',variant:'outline',disabled:busy,onClick:() => act({action:'fix',project:project.id,panel:panelId,mode:'regenerate'},'描き直しをキューに入れました')},'描き直す'))))
    }

    function LetterBox({data, act, busy, page, panels}) {
      const {project} = data
      const [instruction,setInstruction] = React.useState('')
      const running = project.task?.state === 'running'
      return h('div',{className:'mg-card'},
        h('h3',null,panels ? 'AI でこのコマの写植・演出' : 'AI でこのページの写植・演出'),
        h('p',{className:'mg-meta'},'吹き出し・台詞・書き文字・効果線・漫符は絵に描かせず、AI（Claude Code）が絵を見て層として置きます。置いた後は下の画面で動かして直せます。'),
        h('textarea',{rows:2,value:instruction,disabled:busy || running,placeholder:'指示（任意）例: ダンッは画面を横切るほど大きく。吹き出しは鷹野の頭の左に',
          onChange:e => setInstruction(e.target.value)}),
        h('div',{className:'mg-actions'},h(Button,{size:'sm',variant:'primary',disabled:busy || running,
          onClick:() => act({action:'letter',project:project.id,page,panels,instruction,service:'claude'},'AI に頼みました（1 ページ 1〜3 分）')},running ? 'AI が作業中' : 'AI に置かせる')))
    }

    /** One panel large: picture, overlay, and draggable handles for each layer; the selected layer's fields below. */
    function LayerEditor({data, page, placed, act, busy}) {
      const {project} = data
      const overlays = useOverlays(project, page.number)
      const o = overlays[placed.id]
      const catalog = useCatalog()
      const [items,setItems] = React.useState(null)   // local copy while editing
      const [sel,setSel] = React.useState(-1)
      const [drag,setDrag] = React.useState(null)
      const box = React.useRef(null)
      React.useEffect(() => { setItems(null); setSel(-1) }, [placed.id])
      const layers = items ?? o?.layers ?? []
      const dirty = items !== null
      const file = adoptedFile(project, placed.id)
      const update = (i, patch) => setItems(layers.map((l, k) => k === i ? {...l,...patch} : l))
      const save = async next => { if (await act({action:'layers',project:project.id,panel:placed.id,items:next ?? layers}, '保存しました')) setItems(null) }
      function onDown(e, i, part) {
        e.preventDefault(); e.stopPropagation(); setSel(i)
        const rect = box.current.getBoundingClientRect()
        setDrag({i, part, x:e.clientX, y:e.clientY, w:rect.width, h:rect.height, start:layers[i]})
        e.currentTarget.setPointerCapture?.(e.pointerId)
      }
      function onMove(e) {
        if (!drag) return
        const dx = (e.clientX - drag.x) / drag.w, dy = (e.clientY - drag.y) / drag.h
        const l = drag.start, r3 = v => Math.round(v * 1000) / 1000
        if (drag.part === 'tail') update(drag.i, {tail:[r3((l.tail?.[0] ?? 0.5) + dx), r3((l.tail?.[1] ?? 0.5) + dy)]})
        else if (l.type === 'effect') update(drag.i, {cx:r3(l.cx + dx), cy:r3(l.cy + dy)})
        else update(drag.i, {x:r3(l.x + dx), y:r3(l.y + dy)})
      }
      function onUp() { if (drag) { setDrag(null); if (items) save(items) } }
      const add = type => {
        const base = {balloon:{type,style:'speech',lines:['台詞'],x:0.4,y:0.05,w:0.15,h:0.3},sfx:{type,text:'ドンッ',style:'impact',x:0.1,y:0.1,size:0.14,vertical:true,rotate:0},
          text:{type,text:'文字',x:0.1,y:0.8,size:0.04,vertical:false},effect:{type,name:'focus_lines',cx:0.5,cy:0.5,r:0.3,angle:0,density:1},
          mark:{type,name:'sweat_drops',x:0.5,y:0.1,size:0.08,rotate:0}}[type]
        const next = [...layers, base]; setItems(next); setSel(next.length - 1)
      }
      const l = layers[sel]
      const field = (label, key, attrs = {}) => h('label',null,h('span',null,label),h('input',{type:'number',step:0.01,value:l[key] ?? '',...attrs,
        onChange:e => update(sel, {[key]:e.target.value === '' ? undefined : Number(e.target.value)})}))
      return h('div',{className:'mg-card mg-layers'},
        h('div',{className:'mg-stagewrap'},h('div',{className:'mg-stage',ref:box,style:{aspectRatio:String(placed.aspect)},onPointerMove:onMove,onPointerUp:onUp,onPointerCancel:onUp,
            onPointerDown:() => setSel(-1)},
          file ? h('img',{className:'mg-stage-pic',src:fileUrl(project.id, file),alt:''}) : h('span',{className:'mg-beat'},'絵がまだありません'),
          o && !dirty && h('img',{className:'mg-stage-over',src:svgUrl(o.svg),alt:''}),
          o && dirty && h('img',{className:'mg-stage-over is-stale',src:svgUrl(o.svg),alt:''}),
          (o?.boxes || []).map((b, i) => b && layers[i] && h('div',{key:i,className:'mg-handle' + (sel === i ? ' is-selected' : ''),title:LAYER_LABEL[layers[i].type],
            style:(() => {
              const s0 = o.layers[i] || {}, li = layers[i]
              const dx = li.type === 'effect' ? (li.cx - (s0.cx ?? li.cx)) : (li.x - (s0.x ?? li.x)), dy = li.type === 'effect' ? (li.cy - (s0.cy ?? li.cy)) : (li.y - (s0.y ?? li.y))
              const [x1, y1, x2, y2] = li.type === 'effect' ? [li.cx - 0.04, li.cy - 0.04, li.cx + 0.04, li.cy + 0.04] : [b[0] + (dx || 0), b[1] + (dy || 0), b[2] + (dx || 0), b[3] + (dy || 0)]
              return {left:(x1 * 100) + '%',top:(y1 * 100) + '%',width:((x2 - x1) * 100) + '%',height:((y2 - y1) * 100) + '%'}
            })(),
            onPointerDown:e => onDown(e, i, 'body')})),
          l?.type === 'balloon' && l.tail && h('div',{className:'mg-tail',style:{left:(l.tail[0] * 100) + '%',top:(l.tail[1] * 100) + '%'},title:'しっぽの先',
            onPointerDown:e => onDown(e, sel, 'tail')}))),
        h('div',{className:'mg-row'},
          h('span',{className:'mg-meta'},o ? (o.source === 'ai' ? 'AI が置いた層' : o.source === 'edited' ? '手で直した層' : 'ネームからの自動配置') + ' · ' + layers.length + ' 個' : '読み込み中…'),
          ['balloon','sfx','text','effect','mark'].map(t => h(Button,{key:t,size:'sm',variant:'outline',disabled:busy,onClick:() => add(t)},'＋' + LAYER_LABEL[t])),
          dirty && h(Button,{size:'sm',variant:'primary',disabled:busy,onClick:() => save()},'保存'),
          o?.source !== 'auto' && h(Button,{size:'sm',variant:'outline',disabled:busy,onClick:() => confirm('このコマの層を、ネームからの自動配置に戻しますか？') &&
            act({action:'layers',project:project.id,panel:placed.id,reset:true},'自動配置に戻しました').then(() => setItems(null))},'自動配置に戻す')),
        h('p',{className:'mg-meta'},'枠をドラッグで移動（離すと保存）。吹き出しを選ぶと、しっぽの先（丸）も動かせます。'),
        l && h('div',{className:'mg-layer-form'},
          h('strong',null,LAYER_LABEL[l.type]),
          l.type === 'balloon' && h(React.Fragment,null,
            h('label',null,h('span',null,'形'),h('select',{value:l.style,onChange:e => update(sel,{style:e.target.value})},BALLOON_STYLES.map(([v, t]) => h('option',{key:v,value:v},t)))),
            h('label',{className:'mg-wide'},h('span',null,'台詞（列は ／ で区切る。右の列から）'),h('input',{value:(l.lines || []).join('／'),onChange:e => update(sel,{lines:e.target.value.split(/[／/]/)})})),
            field('幅', 'w'), field('高さ', 'h'), field('字の大きさ（倍）', 'size'),
            h('label',{className:'mg-check'},h('input',{type:'checkbox',checked:!!l.tail,onChange:e => update(sel,{tail:e.target.checked ? [l.x + l.w / 2, Math.min(1, l.y + l.h + 0.15)] : undefined})}),h('span',null,'しっぽ'))),
          (l.type === 'sfx' || l.type === 'text') && h(React.Fragment,null,
            h('label',{className:'mg-wide'},h('span',null,'文字'),h('input',{value:l.text || '',onChange:e => update(sel,{text:e.target.value})})),
            l.type === 'sfx' && h('label',{className:'mg-wide'},h('span',null,'書き文字のスタイル'),h('select',{value:l.style || 'impact',onChange:e => update(sel,{style:e.target.value})},
              (catalog?.styles || []).map(s => h('option',{key:s.key,value:s.key},s.ja + ' — ' + s.use)))),
            field('大きさ（コマの幅比）', 'size'), l.type === 'sfx' && field('回転（度）', 'rotate', {step:1}),
            h('label',{className:'mg-check'},h('input',{type:'checkbox',checked:l.vertical !== false && (l.type === 'sfx' || !!l.vertical),onChange:e => update(sel,{vertical:e.target.checked})}),h('span',null,'縦書き')),
            l.type === 'sfx' && h('label',{className:'mg-check'},h('input',{type:'checkbox',checked:l.color === 'white',onChange:e => update(sel,{color:e.target.checked ? 'white' : undefined})}),h('span',null,'白抜き（ベタの上）'))),
          l.type === 'effect' && h(React.Fragment,null,
            h('label',{className:'mg-wide'},h('span',null,'効果'),h('select',{value:l.name,onChange:e => update(sel,{name:e.target.value})},(catalog?.effects || [l.name]).map(n => h('option',{key:n,value:n},n)))),
            field('中心を空ける半径', 'r'), field('密度', 'density', {step:0.1}), field('向き（度）', 'angle', {step:5})),
          l.type === 'mark' && h(React.Fragment,null,
            h('label',{className:'mg-wide'},h('span',null,'漫符'),h('select',{value:l.name,onChange:e => update(sel,{name:e.target.value})},(catalog?.marks || [l.name]).map(n => h('option',{key:n,value:n},n)))),
            field('大きさ', 'size'), field('回転（度）', 'rotate', {step:5})),
          h('div',{className:'mg-actions mg-wide'},
            h(Button,{size:'sm',variant:'primary',disabled:busy || !dirty,onClick:() => save()},'反映して保存'),
            h(Button,{size:'sm',variant:'outline',disabled:busy,onClick:() => { const next = layers.filter((_, k) => k !== sel); setSel(-1); save(next) }},'この層を消す'))))
    }

    function PanelView({data, number, panelId, act, busy, messages, go}) {
      const {project, analysis} = data
      const page = analysis.pages.find(p => p.number === number)
      const placed = page?.panels.find(p => p.id === panelId)
      const name = project.name.pages.find(p => p.number === number)?.panels.find(p => p.id === panelId)
      const [gen,setGen] = React.useState({engine:'anima',count:2,extra:''})
      if (!page || !placed || !name) return h('div',{className:'mg-body'},h('p',{className:'mg-meta'},'コマが見つかりません。'),h(Button,{size:'sm',variant:'outline',onClick:() => go({page:number})},'ページへ'))
      const index = page.panels.indexOf(placed)
      const neighbor = d => page.panels[index + d]?.id
      const withRefs = project.name.characters.filter(c => c.ref && (name.characters || []).some(x => x.id === c.id)).length
      return h('div',{className:'mg-body mg-pagebody'},
        h('div',{className:'mg-pagecol'},
          h('div',{className:'mg-toolbar'},
            h(Button,{size:'sm',variant:'outline',disabled:!neighbor(-1),onClick:() => go({page:number,panel:neighbor(-1)})},'← 前のコマ'),
            h('strong',null,'P' + number + ' · ' + (index + 1) + ' コマ目'),
            h(Button,{size:'sm',variant:'outline',disabled:!neighbor(1),onClick:() => go({page:number,panel:neighbor(1)})},'次のコマ →')),
          h(LayerEditor,{data,page,placed,act,busy}),
          h('div',{className:'mg-card'},
            h('h3',null,'絵を作る'),
            h('div',{className:'mg-grid'},
              h('label',null,h('span',null,'エンジン'),h('select',{value:gen.engine,onChange:e => setGen({...gen,engine:e.target.value,count:e.target.value === 'anima' ? 2 : 1})},
                h('option',{value:'anima'},'Anima（ネイティブ・LoRA 対応）'),h('option',{value:'ph3'},'P-H3（人物の設定画から参照生成' + (withRefs ? '・' + withRefs + ' 人' : '・設定画なし') + '）'))),
              h('label',null,h('span',null,'枚数'),h('input',{type:'number',min:1,max:8,value:gen.count,onChange:e => setGen({...gen,count:e.target.value})}))),
            h('label',null,h('span',null,'今回だけ足すプロンプト（英語・任意）'),h('input',{value:gen.extra,onChange:e => setGen({...gen,extra:e.target.value})})),
            h('div',{className:'mg-actions'},h(Button,{size:'sm',variant:'primary',disabled:busy,
              onClick:() => act({action:'generate',project:project.id,panels:[panelId],engine:gen.engine,count:Number(gen.count),extra:gen.extra},'キューに入れました')},'生成')),
            h('details',null,h('summary',{className:'mg-meta'},'このコマのプロンプト（ネームから自動で作る）'),h('p',{className:'mg-prompt'},gen.engine === 'anima'
              ? [project.render.prefix, placed.anima].filter(Boolean).join(', ') + '\n\nネガティブ: ' + [project.render.negative, placed.avoid].filter(Boolean).join(', ')
              : placed.prompt)),
            messages),
          h(QaResult,{data,panelId,act,busy}),
          h('div',{className:'mg-card'},h('h3',null,'候補と修正'),h(Takes,{data,panelId,act,busy}))),
        h('div',{className:'mg-side'},h(LetterBox,{data,act,busy,page:number,panels:[panelId]}),h(PanelEditor,{data,panel:name,act,busy}),
          h('div',{className:'mg-card'},h('h3',null,'検査'),h(Issues,{issues:analysis.issues,where:panelId}))))
    }

    // ---------- root ----------

    function MangaPanel() {
      const [route,setRoute] = React.useState(() => store.read('route', {}))
      const go = next => {
        const value = next === null ? {} : {project:route.project,...next}
        setRoute(value); store.write('route', value)
        document.querySelector('.mg-scroll')?.scrollTo?.(0, 0)
      }
      return h('section',{className:'mg-root'},
        route.project ? h(ProjectScreen,{key:route.project,route,go}) :
          h(React.Fragment,null,h('header',{className:'mg-head'},h('h1',null,h(Icon),'マンガ')),
            h('div',{className:'mg-scroll'},h(ProjectList,{open:id => go({project:id})}))))
    }

    function ProjectScreen({route, go}) {
      const {data, error, reload} = useProject(route.project)
      const {busy, act, messages} = useAction(reload)
      const title = data?.project.name.meta.title || '…'
      const crumbs = [h('button',{key:'top',type:'button',className:'mg-link',onClick:() => go(null)},'マンガ'),
        ' › ', route.page ? h('button',{key:'p',type:'button',className:'mg-link',onClick:() => go({})},title) : h('span',{key:'p'},title)]
      if (route.page) crumbs.push(' › ', route.panel ? h('button',{key:'pg',type:'button',className:'mg-link',onClick:() => go({page:route.page})},'P' + route.page) : h('span',{key:'pg'},'P' + route.page))
      if (route.panel) crumbs.push(' › ', h('span',{key:'pn'},route.panel))
      return h(React.Fragment,null,
        h('header',{className:'mg-head'},h('h1',null,h(Icon),h('nav',{className:'mg-crumbs','aria-label':'場所'},crumbs)),
          data?.project.task?.state === 'running' && h('span',{className:'mg-pill'},'AI 作業中')),
        h('div',{className:'mg-scroll'},
          error && !data && h('div',{className:'mg-body'},h('p',{className:'mg-error'},error),h(Button,{size:'sm',variant:'outline',onClick:() => go(null)},'一覧へ')),
          !data && !error && h('div',{className:'mg-body'},h('p',{className:'mg-meta'},'読み込み中…')),
          data && (route.panel ? h(PanelView,{data,number:route.page,panelId:route.panel,act,busy,messages,go})
            : route.page ? h(PageView,{data,number:route.page,act,busy,messages,go})
            : h(ProjectView,{data,act,busy,messages,go}))))
    }

    function Icon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('rect',{x:3.5,y:3,width:17,height:18,rx:1.5}),h('path',{d:'M3.5 11h9.5M13 3v18M13 14.5h7.5'}),
        h('path',{d:'M6.5 6.2h3.2a1.4 1.4 0 0 1 0 2.8H8l-1 .9V9a1.4 1.4 0 0 1-.5-2.8z',fill:'currentColor',stroke:'none'}))
    }

    return {inject:['slots'],apply(ctx) {
      const style = document.createElement('style')
      style.textContent = `
        .mg-root{display:flex;flex-direction:column;height:100%;width:100%;min-width:0;min-height:0;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary)}
        .mg-head{display:flex;align-items:center;gap:10px;min-height:28px;padding:10px 16px;border-bottom:1px solid var(--dsw-alias-border-l3)}
        .mg-head h1{display:flex;align-items:center;gap:8px;font:inherit;font-weight:550;margin:0;min-width:0;flex:1}
        .mg-crumbs{min-width:0;overflow-wrap:anywhere;font-weight:550}
        .mg-pill{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--dsw-alias-border-l3);white-space:nowrap}
        .mg-scroll{flex:1;overflow-y:auto;min-height:0}
        .mg-body{box-sizing:border-box;padding:20px 16px;display:grid;gap:16px;align-content:start;max-width:1280px;margin:0 auto;width:100%}
        .mg-pagebody{grid-template-columns:minmax(0,1.25fr) minmax(300px,1fr);align-items:start}
        .mg-pagecol,.mg-side{display:grid;gap:14px;min-width:0;align-content:start}
        .mg-card{box-sizing:border-box;display:grid;gap:12px;padding:16px;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-lg,12px);background:var(--dsw-alias-bg-module-platform);min-width:0}
        .mg-card h2,.mg-card h3{margin:0;font-size:15px;font-weight:600}
        .mg-card label,.mg-card fieldset{display:grid;gap:6px;min-width:0;margin:0;padding:0;border:0}
        .mg-card label>span,.mg-card legend{font-size:12px;font-weight:500;color:var(--dsw-alias-label-secondary);padding:0}
        .mg-root input:not([type]),.mg-root input[type=text],.mg-root input[type=number],.mg-root textarea,.mg-root select{box-sizing:border-box;width:100%;min-width:0;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);padding:7px 9px;font:inherit;color:inherit;background:var(--dsw-alias-bg-base)}
        .mg-root textarea{resize:vertical;line-height:1.5}
        .mg-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
        .mg-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0}
        .mg-row select{flex:1;min-width:140px;width:auto}
        .mg-subrow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px;border:1px solid var(--dsw-alias-border-l3);border-radius:8px}
        .mg-root .mg-subrow select{flex:1 1 90px;width:auto}
        .mg-root .mg-subrow input{flex:1 1 160px;width:auto}
        .mg-root .mg-subrow .mg-wide{flex:1 1 100%;order:3}
        .mg-narrow{width:80px!important;flex-shrink:0}
        .mg-check{display:flex!important;align-items:center;gap:8px!important}
        .mg-check span{font-size:13px!important;color:inherit!important;font-weight:400!important}
        .mg-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
        .mg-start{justify-self:start}
        .mg-meta{font-size:13px;color:var(--dsw-alias-label-secondary);margin:0;overflow-wrap:anywhere}
        .mg-error{color:var(--dsw-alias-state-error-primary);margin:0;overflow-wrap:anywhere;white-space:pre-wrap}
        .mg-toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .mg-toolbar .mg-meta,.mg-toolbar strong{flex:1;text-align:center}
        .mg-toolbar>strong:first-child{text-align:left}
        .mg-link{border:0;background:none;padding:0;color:inherit;font:inherit;cursor:pointer;text-decoration:underline;text-underline-offset:3px}
        .mg-x{border:0;background:none;color:var(--dsw-alias-label-secondary);cursor:pointer;font-size:16px;width:28px;height:28px;border-radius:6px}
        .mg-x:hover{background:var(--dsw-alias-bg-base)}
        .mg-tabs{display:flex;gap:4px;border-bottom:1px solid var(--dsw-alias-border-l3);overflow-x:auto}
        .mg-tabs button{border:0;background:none;color:var(--dsw-alias-label-secondary);font:inherit;padding:8px 12px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
        .mg-tabs button.is-active{color:inherit;border-bottom-color:currentColor;font-weight:550}
        .mg-task{padding:10px 14px;border-radius:var(--dsw-radius-md);border:1px solid var(--dsw-alias-border-l3);display:grid;gap:6px;justify-items:start}
        .mg-task.mg-running{background:var(--dsw-alias-bg-module-platform)}
        .mg-projects{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
        .mg-project{display:grid;gap:4px;text-align:left;padding:10px;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-lg,12px);background:var(--dsw-alias-bg-module-platform);color:inherit;font:inherit;cursor:pointer;min-width:0}
        .mg-project small{color:var(--dsw-alias-label-secondary);font-size:12px;overflow-wrap:anywhere}
        .mg-clamp{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
        .mg-cover{aspect-ratio:1/1.414;border-radius:6px;overflow:hidden;background:#fff;display:grid;place-items:center;border:1px solid var(--dsw-alias-border-l3)}
        .mg-cover img{width:100%;height:100%;object-fit:cover}
        .mg-cover span{font-size:40px;color:#999;font-family:${SERIF}}
        .mg-spreads{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}
        .mg-spread{display:grid;grid-template-columns:1fr 1fr;gap:3px}
        .mg-thumb{display:grid;gap:4px;padding:0;border:0;background:none;color:inherit;font:inherit;cursor:pointer;text-align:center;font-size:12px}
        .mg-thumb.is-blank{visibility:hidden}
        .mg-page{position:relative;width:100%;background:#fff;color:#000;container-type:inline-size;box-shadow:0 0 0 1px var(--dsw-alias-border-l3);overflow:hidden}
        .mg-panel{position:absolute;box-sizing:border-box;border:max(1.5px,.45cqw) solid #000;background:#fff;padding:0;margin:0;overflow:hidden;font:inherit;color:#000;cursor:pointer;display:block}
        div.mg-panel{cursor:default}
        .mg-panel.is-inset{z-index:1}
        .mg-panel.is-selected{outline:3px solid var(--dsw-alias-brand-primary,#3b82f6);outline-offset:1px;z-index:2}
        .mg-panel img{width:100%;height:100%;object-fit:cover;display:block}
        .mg-beat{display:block;padding:4px;font-size:max(9px,1.6cqw);line-height:1.35;color:#777;text-align:left;overflow:hidden;height:100%;box-sizing:border-box}
        .mg-badge{position:absolute;right:4px;bottom:4px;font-size:max(9px,1.4cqw);background:rgba(0,0,0,.7);color:#fff;padding:1px 6px;border-radius:999px}
        .mg-qa{position:absolute;left:4px;top:4px;z-index:4;font-size:max(9px,1.3cqw);background:#c2410c;color:#fff;padding:1px 6px;border-radius:999px}
        .mg-qa.is-ok{background:#15803d}
        .mg-qa-list{margin:0;padding-left:18px;display:grid;gap:4px;font-size:13px}
        .mg-qa-list li.is-ok b{color:var(--dsw-alias-state-success-primary,#15803d)}
        .mg-sheets{display:grid;gap:8px}
        .mg-sheet-list{display:flex;flex-wrap:wrap;gap:8px}
        .mg-sheet-list figure{margin:0;display:grid;gap:4px;width:96px}
        .mg-sheet-list img{width:96px;height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--dsw-alias-border-l3);background:#fff}
        .mg-sheet-list figcaption{display:flex;flex-wrap:wrap;gap:4px}
        .mg-sheet-list figcaption button{border:1px solid var(--dsw-alias-border-l3);background:var(--dsw-alias-bg-base);color:inherit;font:inherit;font-size:11px;border-radius:6px;padding:1px 6px;cursor:pointer}
        .mg-overlay{position:absolute;z-index:3;pointer-events:none;display:block}
        .mg-stagewrap{overflow:hidden;padding:7%;background:#e9e9e9;border-radius:8px}
        .mg-stage{position:relative;width:100%;background:#fff;box-shadow:0 0 0 2px #000;overflow:visible;touch-action:none;user-select:none}
        .mg-stage-pic{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none}
        .mg-stage-over{position:absolute;left:-15%;top:-15%;width:130%;height:130%;pointer-events:none}
        .mg-stage-over.is-stale{opacity:.45}
        .mg-handle{position:absolute;z-index:2;box-sizing:border-box;border:1.5px dashed rgba(59,130,246,.9);border-radius:4px;cursor:move;background:rgba(59,130,246,.06)}
        .mg-handle.is-selected{border-style:solid;border-width:2px;background:rgba(59,130,246,.14)}
        .mg-tail{position:absolute;z-index:3;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;background:#3b82f6;border:2px solid #fff;cursor:crosshair}
        .mg-layer-form{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;align-items:end;padding:10px;border:1px solid var(--dsw-alias-border-l3);border-radius:8px}
        .mg-layer-form>strong,.mg-layer-form .mg-wide{grid-column:1/-1}
        .mg-layouts{display:grid;gap:8px}
        .mg-layout-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px;max-height:420px;overflow-y:auto}
        .mg-layout{display:grid;gap:2px;padding:6px;border:1px solid var(--dsw-alias-border-l3);border-radius:8px;background:var(--dsw-alias-bg-base);color:inherit;font:inherit;cursor:pointer;text-align:left}
        .mg-layout.is-selected{border-color:var(--dsw-alias-brand-primary,currentColor);box-shadow:0 0 0 1px var(--dsw-alias-brand-primary,currentColor)}
        .mg-layout svg{width:100%;height:auto;background:#f6f6f6;color:#111;border-radius:4px}
        .mg-layout small{font-size:11px;line-height:1.3}
        .mg-panel-list{margin:0;padding-left:20px;display:grid;gap:4px;font-size:13px}
        .mg-panel-list li{display:flex;align-items:center;gap:4px}
        .mg-panel-list .mg-link{text-align:left;overflow-wrap:anywhere;flex:1;min-width:0}
        .mg-panel-list small{white-space:nowrap;flex-shrink:0}
        .mg-issues{margin:0;padding-left:18px;display:grid;gap:4px;font-size:13px}
        .mg-issues .mg-FAIL b{color:var(--dsw-alias-state-error-primary)}
        .mg-plan{white-space:pre-wrap;font:inherit;font-size:13px;margin:0;overflow-wrap:anywhere}
        .mg-purposes{margin:0;padding-left:22px;display:grid;gap:6px;font-size:13px}
        .mg-prompt{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0 0;color:var(--dsw-alias-label-secondary)}
        .mg-takes{display:grid;gap:12px}
        .mg-take{display:grid;gap:6px;padding-top:10px;border-top:1px solid var(--dsw-alias-border-l3)}
        .mg-take .mg-link{justify-self:start;font-size:13px}
        .mg-files{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
        .mg-files figure{margin:0;display:grid;gap:4px;align-content:start;padding:4px;border-radius:8px;border:2px solid transparent}
        .mg-files figure.is-adopted{border-color:var(--dsw-alias-brand-primary,#3b82f6)}
        .mg-files img{width:100%;height:auto;display:block;border-radius:4px;background:#fff}
        .mg-files figcaption{display:flex;flex-wrap:wrap;gap:6px;font-size:12px;align-items:center}
        .mg-files figcaption button{border:1px solid var(--dsw-alias-border-l3);background:var(--dsw-alias-bg-base);color:inherit;font:inherit;border-radius:6px;padding:2px 8px;cursor:pointer}
        .mg-files figcaption select{width:auto;padding:2px 4px;font-size:12px}
        .mg-fix{grid-column:1/-1;display:grid;gap:6px}
        .mg-character{display:grid;grid-template-columns:120px minmax(0,1fr);gap:12px;padding-top:12px;border-top:1px solid var(--dsw-alias-border-l3)}
        .mg-fields{display:grid;gap:8px;min-width:0}
        .mg-ref{display:grid;gap:6px;align-content:start}
        .mg-ref img{width:120px;height:160px;object-fit:cover;border-radius:6px;border:1px solid var(--dsw-alias-border-l3)}
        .mg-ref>span{width:120px;height:160px;display:grid;place-items:center;border:1px dashed var(--dsw-alias-border-l3);border-radius:6px;font-size:12px;color:var(--dsw-alias-label-secondary)}
        .mg-file{position:relative;font-size:12px!important;cursor:pointer;text-decoration:underline}
        .mg-file input{position:absolute;inset:0;opacity:0;cursor:pointer}
        .mg-file span{font-size:12px!important;color:inherit!important;overflow-wrap:anywhere}
        details summary{cursor:pointer}
        @media(max-width:900px){.mg-pagebody{grid-template-columns:1fr}}
        @media(max-width:650px){.mg-head{padding:8px 12px}.mg-body{padding:16px 12px}.mg-character{grid-template-columns:1fr}.mg-spreads{grid-template-columns:1fr}}
      `
      document.head.append(style)
      ctx.effect(() => () => style.remove(), 'Manga panel styles')
      ctx.slots.inject('main', () => ctx.slots.register({name:'main',key:'manga-studio'}, MangaPanel))
      ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
        name:'sidebar.panellist',id:'manga-studio',order:25,label:() => 'マンガ'
      }, Icon))
    }}
  }
})
