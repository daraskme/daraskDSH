window.__ModuleLoader__.load({
  id:'dsh-anima-lora-trainer',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const {Button, IconFolderOpenRegular} = require('@deepseek-ai/dsh-client-ui-primitives')
    const API = '/api/darask.animaLora'
    const PURPOSES = [['character','キャラ','一人の人物・キャラの見た目を覚えさせる'],
      ['style','スタイル','絵柄・画風を覚えさせる'],['concept','コンセプト','ポーズ・物・状況などの概念を覚えさせる']]
    const STAGES = {queued:'待機中',starting:'準備中',captioning:'キャプション作成',training:'学習',loading:'モデル読み込み',
      generating:'生成中',done:'完了',error:'エラー',cancelled:'中止'}
    const SIZES = [['832x1216','縦 832×1216'],['1216x832','横 1216×832'],['1024x1024','正方形 1024×1024'],
      ['896x1152','縦 896×1152'],['1152x896','横 1152×896']]
    const imageUrl = (job, file) => API + '?image=' + encodeURIComponent(job.id + '/' + file)

    async function api(body) {
      const res = await fetch(API, body ? {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)} : {})
      const data = await res.json()
      if (!res.ok) throw Error(data.error || 'エラー ' + res.status)
      return data
    }
    const lr = value => value < 0.001 ? value.toExponential().replace('e-', 'e-') : String(value)
    const minutes = ms => ms < 60000 ? Math.max(1, Math.round(ms / 1000)) + ' 秒' : Math.round(ms / 60000) + ' 分'

    function Settings({plan}) {
      const s = plan.settings, a = s.args
      const rows = [
        ['画像', plan.images + ' 枚（キャプションあり ' + plan.captioned + ' 枚）'],
        ['学習量', s.epochs + ' epoch · ' + s.steps + ' step'],
        ['解像度 / rank', s.resolution + 'px · rank ' + s.rank],
        ['学習率', lr(a['learning-rate']) + (a['cross-attn-lr'] !== undefined ? '（cross-attn ' + lr(a['cross-attn-lr']) + '）' : '')],
        ['スケジュール', 'warmup ' + Math.round(a.warmup * 100) + '% · cosine ' + a['lr-cycles'] + ' 回 · timestep ×' + a['timestep-scale']],
        ['出力', plan.comfyName],
      ]
      return h('dl',{className:'dsh-anima-settings'},rows.flatMap(([k,v])=>[h('dt',{key:k},k),h('dd',{key:k+'v'},v)]))
    }

    function GenerateJob({job, cancel}) {
      const r = job.request || {}
      const fraction = r.count ? ((job.done || 0) + (job.stage === 'generating' && job.step ? job.step / job.steps : 0)) / r.count : 0
      const images = job.images || []
      const each = images.length ? images.reduce((sum, image) => sum + image.seconds, 0) / images.length : 0
      return h('section',{className:'dsh-anima-card','aria-live':'polite'},
        h('header',null,h('strong',{className:'dsh-anima-name'},'画像生成 · ' + r.count + ' 枚'),
          h('span',{className:'dsh-anima-stage dsh-anima-'+job.stage},
            job.queued ? STAGES.queued + '（' + job.position + ' 番目）' : STAGES[job.stage] || job.stage)),
        h('p',{className:'dsh-anima-prompt'},r.prompt),
        h('div',{className:'dsh-anima-meta'},r.width + '×' + r.height + ' · ' + r.steps + ' step · CFG ' + r.cfg + ' · seed ' + r.seed +
          (r.count > 1 ? '〜' + (r.seed + r.count - 1) : '') +
          (r.loras?.length ? ' · ' + r.loras.map(l => l.name.replace(/\.safetensors$/,'') + ' ×' + l.strength).join(', ') : '')),
        job.stage === 'generating' && h(React.Fragment,null,h('progress',{max:1,value:fraction}),
          h('div',{className:'dsh-anima-meta'},(job.done || 0) + ' / ' + r.count + ' 枚' + (job.step ? ' · step ' + job.step + ' / ' + job.steps : '') +
            (each ? ' · 1 枚 ' + each.toFixed(1) + ' 秒' : ''))),
        job.message && job.stage !== 'done' && h('p',{className:job.stage === 'error' ? 'dsh-anima-error' : 'dsh-anima-meta',role:job.stage === 'error' ? 'alert' : undefined},job.message),
        images.length > 0 && h('div',{className:'dsh-anima-images'},images.map(image =>
          h('a',{key:image.file,href:imageUrl(job, image.file),target:'_blank',rel:'noopener',title:'seed ' + image.seed},
            h('img',{src:imageUrl(job, image.file),alt:'seed ' + image.seed,loading:'lazy',style:{aspectRatio:r.width + '/' + r.height}})))),
        job.stage === 'done' && h('p',{className:'dsh-anima-meta'},'読み込み ' + Math.round(job.load_seconds || 0) + ' 秒 · 1 枚 ' + each.toFixed(1) + ' 秒'),
        (job.running || job.queued) && h(Button,{variant:'outline',size:'sm',onClick:() => cancel(job.id)},job.running ? '中止' : '取り消す'))
    }

    function Job({job, cancel}) {
      if (job.request?.kind === 'generate') return h(GenerateJob,{job,cancel})
      const fraction = job.total ? job.done / job.total : 0
      const eta = job.stage === 'training' && job.step_ms && job.total ? (job.total - job.done) * job.step_ms : 0
      return h('section',{className:'dsh-anima-card','aria-live':'polite'},
        h('header',null,h('strong',{className:'dsh-anima-name'},job.request?.comfyName?.replace(/^trained\//,'')),
          h('span',{className:'dsh-anima-stage dsh-anima-'+job.stage},
            job.queued ? STAGES.queued + '（' + job.position + ' 番目）' : STAGES[job.stage] || job.stage)),
        h('div',{className:'dsh-anima-meta'},
          (PURPOSES.find(([key]) => key === job.request?.purpose)?.[1] || '') + ' · ' + job.request?.images?.length + ' 枚 · ' +
          job.request?.steps + ' step' + (job.request?.trigger ? ' · ' + job.request.trigger : '') + ' · ' + job.request?.folder),
        (job.stage === 'captioning' || job.stage === 'training') && h(React.Fragment,null,
          h('progress',{max:1,value:fraction}),
          h('div',{className:'dsh-anima-meta'},
            job.stage === 'captioning' ? 'キャプション ' + job.done + ' / ' + job.total :
              'step ' + job.done + ' / ' + job.total + (job.loss !== undefined ? ' · loss ' + job.loss.toFixed(4) : '') + (eta ? ' · 残り約 ' + minutes(eta) : ''))),
        job.message && h('p',{className:job.stage === 'error' ? 'dsh-anima-error' : 'dsh-anima-meta',role:job.stage === 'error' ? 'alert' : undefined},job.message),
        job.last_caption && job.stage === 'captioning' && h('blockquote',null,job.last_caption),
        job.stage === 'done' && job.result && h('p',null,
          'ComfyUI の LoRA 読み込みで ', h('code',null,job.request?.comfyName), ' を選べます。',
          ' 学習 ' + Math.round(job.result.training_seconds) + ' 秒 · ' + job.result.steps + ' step'),
        job.backup && h('p',{className:'dsh-anima-meta'},'元のキャプションの退避先: ' + job.backup),
        (job.running || job.queued) && h(Button,{variant:'outline',size:'sm',onClick:() => cancel(job.id)},job.running ? '中止' : '取り消す'))
    }

    function FolderBrowser({start, choose, close}) {
      const [view,setView] = React.useState(null)
      const [error,setError] = React.useState('')
      const open = path => api({action:'list',path}).then(data => { setView(data); setError('') }).catch(e => setError(e.message))
      React.useEffect(() => { open(start || undefined) }, [])
      const desktop = () => fetch('/api/darask.pickWorkspace',{method:'POST',headers:{'content-type':'application/json'},body:'{}'})
        .then(async res => { const data = await res.json(); if (!res.ok) throw Error(data.error); if (data.path) choose(data.path) })
        .catch(e => setError(e.message))
      return h('div',{className:'dsh-anima-browser',role:'group','aria-label':'フォルダの選択'},
        h('div',{className:'dsh-anima-row'},
          h('code',{className:'dsh-anima-path'},view?.path || '…'),
          h(Button,{type:'button',size:'sm',variant:'outline',onClick:desktop},'Ubuntu で選択'),
          h(Button,{type:'button',size:'sm',variant:'outline',onClick:close},'閉じる')),
        view && h('ul',null,
          view.parent && h('li',null,h('button',{type:'button',onClick:() => open(view.parent)},'↑ 上のフォルダ')),
          view.dirs.map(dir => h('li',{key:dir.path},h('button',{type:'button',onClick:() => open(dir.path)},
            h(IconFolderOpenRegular,{size:14}),h('span',null,dir.name),dir.images > 0 && h('small',null,dir.images + ' 枚'))))),
        view && h(Button,{type:'button',variant:'primary',size:'sm',disabled:!view.images,onClick:() => choose(view.path)},
          view.images ? 'このフォルダを使う（画像 ' + view.images + ' 枚）' : 'このフォルダに画像はありません'),
        error && h('p',{className:'dsh-anima-error',role:'alert'},error))
    }

    function useQueue() {
      const [jobs,setJobs] = React.useState([])
      React.useEffect(() => {
        let stop = false, timer
        const poll = async () => {
          try { const data = await api(); if (!stop) setJobs(data.jobs) } catch {}
          if (!stop) timer = setTimeout(poll, 2000)
        }
        poll()
        return () => { stop = true; clearTimeout(timer) }
      }, [])
      return [jobs, setJobs]
    }

    function Queue({jobs, setJobs, setError}) {
      const cancel = id => api({action:'cancel',id}).then(data => setJobs(data.jobs)).catch(e => setError(e.message))
      const clear = () => api({action:'clear'}).then(data => setJobs(data.jobs)).catch(e => setError(e.message))
      const active = jobs.filter(job => job.running || job.queued).length
      const finished = jobs.length - active
      return h(React.Fragment,null,
        jobs.length > 0 && h('div',{className:'dsh-anima-queue-head'},
          h('strong',null,'キュー（学習と生成で共通）'),h('span',{className:'dsh-anima-meta'},'実行中・待機中 ' + active + ' 件'),
          finished > 0 && h(Button,{variant:'outline',size:'sm',onClick:clear},'終了したものを消す')),
        jobs.map(job => h(Job,{key:job.id,job,cancel})))
    }

    function GeneratePanel() {
      const [form,setForm] = React.useState({prompt:'',negative:'',size:'832x1216',steps:34,cfg:4.5,seed:'',count:1,loras:[]})
      const [available,setAvailable] = React.useState([])
      const [jobs,setJobs] = useQueue()
      const [notice,setNotice] = React.useState('')
      const [error,setError] = React.useState('')
      const [busy,setBusy] = React.useState(false)
      const set = patch => setForm(previous => ({...previous,...patch}))
      React.useEffect(() => { api({action:'loras'}).then(data => setAvailable(data.loras)).catch(e => setError(e.message)) }, [])
      const setLora = (index, patch) => set({loras:form.loras.map((item, i) => i === index ? {...item,...patch} : item)})

      async function submit(e) {
        e.preventDefault()
        setBusy(true); setError(''); setNotice('')
        try {
          const [width, height] = form.size.split('x').map(Number)
          const data = await api({action:'generate',prompt:form.prompt,negative:form.negative,width,height,steps:Number(form.steps),
            cfg:Number(form.cfg),count:Number(form.count),seed:form.seed === '' ? null : Number(form.seed),
            loras:form.loras.filter(item => item.name).map(item => ({name:item.name,strength:Number(item.strength)}))})
          setJobs(data.jobs)
          setNotice('キューに追加しました')
        } catch (e) { setError(e.message) }
        setBusy(false)
      }
      const active = jobs.filter(job => job.running || job.queued).length
      return h('section',{className:'dsh-anima-panel'},
        h('header',{className:'dsh-anima-head'},h('h1',null,h(GenerateIcon),'Anima 生成')),
        h('div',{className:'dsh-anima-body'},
          h('form',{className:'dsh-anima-card',onSubmit:submit},
            h('label',null,h('span',null,'プロンプト'),
              h('textarea',{value:form.prompt,rows:4,required:true,disabled:busy,placeholder:'例: @darask, (red scarf:1.3), a young woman ...',
                onChange:e => set({prompt:e.target.value}),onKeyDown:e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) e.currentTarget.form.requestSubmit() }})),
            h('label',null,h('span',null,'ネガティブプロンプト'),
              h('textarea',{value:form.negative,rows:2,disabled:busy,placeholder:'空欄可',onChange:e => set({negative:e.target.value})})),
            h('fieldset',{disabled:busy},h('legend',null,'LoRA'),
              form.loras.map((item, index) => h('div',{key:index,className:'dsh-anima-row'},
                h('select',{value:item.name,'aria-label':'LoRA',onChange:e => setLora(index,{name:e.target.value})},
                  h('option',{value:''},'選択してください'),available.map(name => h('option',{key:name,value:name},name.replace(/\.safetensors$/,'')))),
                h('input',{type:'number',step:0.05,min:-2,max:2,value:item.strength,className:'dsh-anima-narrow','aria-label':'強さ',
                  onChange:e => setLora(index,{strength:e.target.value})}),
                h(Button,{type:'button',size:'sm',variant:'outline',onClick:() => set({loras:form.loras.filter((_, i) => i !== index)})},'外す'))),
              form.loras.length < 4 && h(Button,{type:'button',size:'sm',variant:'outline',className:'dsh-anima-start',
                disabled:!available.length,onClick:() => set({loras:[...form.loras,{name:'',strength:1}]})},
                available.length ? 'LoRA を追加' : '使える Anima LoRA がありません')),
            h('div',{className:'dsh-anima-grid'},
              h('label',null,h('span',null,'サイズ'),h('select',{value:form.size,disabled:busy,onChange:e => set({size:e.target.value})},
                SIZES.map(([value,label]) => h('option',{key:value,value},label)))),
              h('label',null,h('span',null,'枚数'),h('input',{type:'number',min:1,max:16,value:form.count,disabled:busy,onChange:e => set({count:e.target.value})})),
              h('label',null,h('span',null,'シード'),h('input',{type:'number',min:0,value:form.seed,placeholder:'ランダム',disabled:busy,onChange:e => set({seed:e.target.value})})),
              h('label',null,h('span',null,'step'),h('input',{type:'number',min:1,max:100,value:form.steps,disabled:busy,onChange:e => set({steps:e.target.value})})),
              h('label',null,h('span',null,'CFG'),h('input',{type:'number',min:1,max:20,step:0.1,value:form.cfg,disabled:busy,onChange:e => set({cfg:e.target.value})}))),
            h('div',{className:'dsh-anima-actions'},
              h(Button,{type:'submit',variant:'primary',disabled:busy || !form.prompt.trim()},
                active ? 'キューに追加（実行中・待機中 ' + active + ' 件）' : '生成')),
            h('p',{className:'dsh-anima-meta'},'er_sde · simple · Anima base v1.0。ComfyUI と同じ設定なら同じ画像になります。'),
            notice && h('p',{className:'dsh-anima-meta',role:'status'},notice),
            error && h('p',{className:'dsh-anima-error',role:'alert'},error)),
          h(Queue,{jobs,setJobs,setError})))
    }

    function Panel() {
      const [form,setForm] = React.useState({folder:'',purpose:'style',trigger:'',name:'',overwrite:true})
      const [plan,setPlan] = React.useState(null)
      const [jobs,setJobs] = useQueue()
      const [notice,setNotice] = React.useState('')
      const [error,setError] = React.useState('')
      const [busy,setBusy] = React.useState(false)
      const [picking,setPicking] = React.useState(false)
      const set = patch => { setForm(previous => ({...previous,...patch})); setPlan(null) }

      async function run(action) {
        setBusy(true); setError(''); setNotice('')
        try {
          const data = await api({action,...form})
          if (action === 'plan') setPlan(data)
          else { setJobs(data.jobs); setPlan(null); setNotice('キューに追加しました: ' + plan.comfyName); setForm(previous => ({...previous,name:''})) }
        } catch (e) { setError(e.message) }
        setBusy(false)
      }
      const active = jobs.filter(job => job.running || job.queued).length

      const disabled = busy
      return h('section',{className:'dsh-anima-panel'},
        h('header',{className:'dsh-anima-head'},h('h1',null,h(Icon),'Anima LoRA 学習')),
        h('div',{className:'dsh-anima-body'},
          h('form',{className:'dsh-anima-card',onSubmit:e => { e.preventDefault(); run('plan') }},
            h('label',null,h('span',null,'画像フォルダ'),
              h('div',{className:'dsh-anima-row'},
                h('input',{value:form.folder,placeholder:'/path/to/images',disabled,required:true,onChange:e => set({folder:e.target.value})}),
                h(Button,{type:'button',variant:'outline',disabled,className:'dsh-anima-nowrap',icon:h(IconFolderOpenRegular,{size:15}),onClick:() => setPicking(!picking)},'選択…')),
              picking && h(FolderBrowser,{start:form.folder.trim(),close:() => setPicking(false),
                choose:path => { set({folder:path}); setPicking(false) }})),
            h('fieldset',{disabled},h('legend',null,'目的'),
              h('div',{className:'dsh-anima-purposes'},PURPOSES.map(([key,label,hint]) =>
                h('label',{key,className:form.purpose === key ? 'is-selected' : ''},
                  h('input',{type:'radio',name:'purpose',value:key,checked:form.purpose === key,onChange:() => set({purpose:key})}),
                  h('strong',null,label),h('small',null,hint))))),
            h('label',null,h('span',null,'トリガーワード'),
              h('input',{value:form.trigger,placeholder:'例: @darask（空欄可）',maxLength:100,disabled,onChange:e => set({trigger:e.target.value})})),
            h('label',null,h('span',null,'出力名'),
              h('input',{value:form.name,placeholder:'空欄ならトリガーワードかフォルダ名',disabled,onChange:e => set({name:e.target.value})})),
            h('label',{className:'dsh-anima-check'},
              h('input',{type:'checkbox',checked:form.overwrite,disabled,onChange:e => set({overwrite:e.target.checked})}),
              h('span',null,'既存のキャプションも Gemma で作り直す（元のファイルはフォルダ内に退避）')),
            h('div',{className:'dsh-anima-actions'},
              h(Button,{type:'submit',variant:plan ? 'outline' : 'primary',disabled:disabled || !form.folder.trim()},'設定を確認'),
              plan && h(Button,{type:'button',variant:'primary',disabled:disabled || plan.exists,onClick:() => run('start')},
                active ? 'キューに追加（実行中・待機中 ' + active + ' 件）' : 'キャプション作成と学習を開始')),
            plan && h(Settings,{plan}),
            plan?.exists && h('p',{className:'dsh-anima-error',role:'alert'},plan.queued ? 'キューに同じ出力名の LoRA があります。出力名を変えてください。' : '同じ名前の LoRA があります。出力名を変えてください。'),
            notice && h('p',{className:'dsh-anima-meta',role:'status'},notice),
            error && h('p',{className:'dsh-anima-error',role:'alert'},error)),
          h(Queue,{jobs,setJobs,setError})))
    }

    function Icon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('rect',{x:3.5,y:4.5,width:17,height:15,rx:2.5}),
        h('path',{d:'M3.5 15.5l4.5-4.5 3.5 3.5 2.5-2.5 6.5 6.5'}),
        h('path',{d:'M16.5 6.8l.6 1.6 1.6.6-1.6.6-.6 1.6-.6-1.6-1.6-.6 1.6-.6z',fill:'currentColor',stroke:'none'}))
    }

    function GenerateIcon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('path',{d:'M4 20l9.5-9.5'}),h('path',{d:'M12 9l3 3'}),
        h('path',{d:'M16.5 3.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z',fill:'currentColor',stroke:'none'}),
        h('path',{d:'M19.5 11.5l.5 1.2 1.2.5-1.2.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z',fill:'currentColor',stroke:'none'}),
        h('path',{d:'M8.5 3.5l.5 1.2 1.2.5-1.2.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z',fill:'currentColor',stroke:'none'}))
    }

    return {inject:['slots'],apply(ctx) {
      const style = document.createElement('style')
      style.textContent = `
        .dsh-anima-panel{display:flex;flex-direction:column;height:100%;width:100%;min-width:0;min-height:0;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary)}
        .dsh-anima-head{display:flex;align-items:center;min-height:28px;padding:10px 16px;border-bottom:1px solid var(--dsw-alias-border-l3)}
        .dsh-anima-head h1{display:flex;align-items:center;gap:8px;font:inherit;font-weight:550;margin:0}
        .dsh-anima-body{flex:1;overflow-y:auto;padding:20px 16px;display:grid;gap:16px;align-content:start;justify-items:center}
        .dsh-anima-card{box-sizing:border-box;width:min(640px,100%);display:grid;gap:14px;padding:16px;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-lg,12px);background:var(--dsw-alias-bg-module-platform)}
        .dsh-anima-card label,.dsh-anima-card fieldset{display:grid;gap:6px;min-width:0;margin:0;padding:0;border:0}
        .dsh-anima-card label>span,.dsh-anima-card legend{font-size:12px;font-weight:500;color:var(--dsw-alias-label-secondary);padding:0}
        .dsh-anima-card input:not([type]),.dsh-anima-card input[type=text],.dsh-anima-card input[type=number],.dsh-anima-card textarea,.dsh-anima-card select{box-sizing:border-box;width:100%;min-width:0;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);padding:8px 10px;font:inherit;color:inherit;background:var(--dsw-alias-bg-base)}
        .dsh-anima-row{display:flex;gap:8px;min-width:0;align-items:center}
        .dsh-anima-nowrap{white-space:nowrap;flex-shrink:0}
        .dsh-anima-browser{display:grid;gap:8px;padding:10px;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);background:var(--dsw-alias-bg-base)}
        .dsh-anima-path{flex:1;min-width:0;font-size:12px;overflow-wrap:anywhere}
        .dsh-anima-browser ul{list-style:none;margin:0;padding:0;max-height:260px;overflow-y:auto;display:grid;gap:2px}
        .dsh-anima-browser li button{display:flex;align-items:center;gap:8px;width:100%;padding:6px 8px;border:0;border-radius:6px;background:none;color:inherit;font:inherit;text-align:left;cursor:pointer}
        .dsh-anima-browser li button:hover{background:var(--dsw-alias-bg-module-platform)}
        .dsh-anima-browser li span{flex:1;min-width:0;overflow-wrap:anywhere}
        .dsh-anima-browser li small{color:var(--dsw-alias-label-secondary)}
        .dsh-anima-purposes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
        .dsh-anima-purposes label{display:grid;gap:2px;padding:10px;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);cursor:pointer;background:var(--dsw-alias-bg-base)}
        .dsh-anima-purposes label.is-selected{border-color:var(--dsw-alias-brand-primary,currentColor);box-shadow:0 0 0 1px var(--dsw-alias-brand-primary,currentColor)}
        .dsh-anima-purposes input{position:absolute;opacity:0;pointer-events:none}
        .dsh-anima-purposes small{color:var(--dsw-alias-label-secondary);font-size:12px}
        .dsh-anima-check{display:flex!important;align-items:flex-start;gap:8px!important}
        .dsh-anima-check span{font-size:13px!important;color:inherit!important;font-weight:400!important}
        .dsh-anima-actions{display:flex;flex-wrap:wrap;gap:8px}
        .dsh-anima-settings{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0;font-size:13px}
        .dsh-anima-settings dt{color:var(--dsw-alias-label-secondary)}
        .dsh-anima-settings dd{margin:0;overflow-wrap:anywhere}
        .dsh-anima-card header{display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0}
        .dsh-anima-name{min-width:0;overflow-wrap:anywhere}
        .dsh-anima-stage{flex-shrink:0;white-space:nowrap}
        .dsh-anima-queue-head{box-sizing:border-box;width:min(640px,100%);display:flex;align-items:center;gap:10px}
        .dsh-anima-queue-head .dsh-anima-meta{flex:1}
        .dsh-anima-card>button{justify-self:start}
        .dsh-anima-stage{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--dsw-alias-border-l3)}
        .dsh-anima-done{color:var(--dsw-alias-state-success-primary,inherit)}
        .dsh-anima-error{color:var(--dsw-alias-state-error-primary);margin:0;overflow-wrap:anywhere}
        .dsh-anima-meta{font-size:13px;color:var(--dsw-alias-label-secondary);margin:0;overflow-wrap:anywhere}
        .dsh-anima-card progress{width:100%;height:8px}
        .dsh-anima-card blockquote{margin:0;padding:8px 10px;border-left:3px solid var(--dsw-alias-border-l3);font-size:12px;color:var(--dsw-alias-label-secondary);overflow-wrap:anywhere}
        .dsh-anima-card p{margin:0}
        .dsh-anima-card textarea{resize:vertical;line-height:1.5}
        .dsh-anima-grid{display:grid;grid-template-columns:2fr repeat(4,minmax(0,1fr));gap:8px}
        .dsh-anima-narrow{width:80px!important;flex-shrink:0}
        .dsh-anima-row select{flex:1;min-width:0}
        .dsh-anima-start{justify-self:start}
        .dsh-anima-prompt{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
        .dsh-anima-images{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
        .dsh-anima-images img{display:block;width:100%;height:auto;border-radius:var(--dsw-radius-md);background:var(--dsw-alias-bg-base)}
        @media(max-width:650px){.dsh-anima-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.dsh-anima-grid label:first-child{grid-column:1/-1}}
        @media(max-width:650px){.dsh-anima-purposes{grid-template-columns:1fr}.dsh-anima-head{padding:8px 12px}.dsh-anima-body{padding:16px 12px}}
      `
      document.head.append(style)
      ctx.effect(() => () => style.remove(), 'Anima LoRA panel styles')
      ctx.slots.inject('main', () => ctx.slots.register({name:'main',key:'anima-lora'}, Panel))
      ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
        name:'sidebar.panellist',id:'anima-lora',order:21,label:() => 'Anima LoRA'
      }, Icon))
      ctx.slots.inject('main', () => ctx.slots.register({name:'main',key:'anima-generate'}, GeneratePanel))
      ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
        name:'sidebar.panellist',id:'anima-generate',order:22,label:() => 'Anima 生成'
      }, GenerateIcon))
    }}
  }
})
