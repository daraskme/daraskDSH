window.__ModuleLoader__.load({
  id:'dsh-media-shared',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const {Button} = require('@deepseek-ai/dsh-client-ui-primitives')
    const API = '/api/darask.animaLora'
    const PURPOSES = [['character','キャラ','一人の人物・キャラの見た目を覚えさせる'],
      ['style','スタイル','絵柄・画風を覚えさせる'],['concept','コンセプト','ポーズ・物・状況などの概念を覚えさせる']]
    const STAGES = {queued:'待機中',starting:'準備中',captioning:'キャプション作成',training:'学習',loading:'モデル読み込み',
      generating:'生成中',decoding:'動画を作成',done:'完了',error:'エラー',cancelled:'中止'}
    const imageUrl = (job, file) => API + '?image=' + encodeURIComponent(job.id + '/' + file)
    const IMAGE_SIZES = [['2912x2912','正方形 2912×2912（標準）'],['3872x2176','横 3872×2176'],['2176x3872','縦 2176×3872'],
      ['2048x2048','正方形 2048×2048'],['1024x1024','正方形 1024×1024（試し用）']]

    async function api(body) {
      const res = await fetch(API, body ? {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)} : {})
      const data = await res.json()
      if (!res.ok) throw Error(data.error || 'エラー ' + res.status)
      return data
    }
    // the desktop folder dialog only makes sense on the machine itself, not through studio.darask.me
    const LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)
    const uploadUrl = token => API + '?upload=' + encodeURIComponent(token)
    const tokenOf = path => String(path).split('/').pop()
    const toBase64 = file => new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result).split(',')[1] || '')
      reader.onerror = () => reject(reader.error)
      reader.readAsDataURL(file)
    })
    const IMAGE_MODES = [['t2i','テキストから'],['reference','参照画像から生成']]
    const MODE_LABEL = {i2v:'画像から',ref:'参照画像から',edit:'編集',reference:'参照から生成'}

    /** Picks pictures, sends each one to the server (works the same locally and through the remote gateway) and
     *  keeps their tokens. Drop files on it or use the file picker (phones offer the camera and photo library). */
    function Pictures({label, hint, max, value, onChange, disabled, captions}) {
      const [busy,setBusy] = React.useState(false)
      const [error,setError] = React.useState('')
      async function add(files) {
        setBusy(true); setError('')
        try {
          let next = [...value]
          for (const file of [...files].slice(0, max - value.length)) {
            if (file.size > 20 << 20) throw Error(file.name + ' は 20 MB を超えています')
            const {token} = await api({action:'upload',name:file.name,data:await toBase64(file)})
            next = [...next, token]
            onChange(next)
          }
        } catch (e) { setError(e.message) }
        setBusy(false)
      }
      return h('fieldset',{className:'dsh-anima-pictures',disabled:disabled || busy,
          onDragOver:e => e.preventDefault(),onDrop:e => { e.preventDefault(); if (!disabled && !busy) add(e.dataTransfer.files) }},
        h('legend',null,label),
        h('div',{className:'dsh-anima-thumbs'},
          value.map((token, index) => h('figure',{key:token},
            h('img',{src:uploadUrl(token),alt:(captions?.(index) || label) + ' ' + (index + 1)}),
            h('figcaption',null,captions ? captions(index) : max > 1 ? '<Picture ' + (index + 1) + '>' : ''),
            h('button',{type:'button','aria-label':'外す',onClick:() => onChange(value.filter(item => item !== token))},'×'))),
          value.length < max && h('label',{className:'dsh-anima-add'},
            h('input',{type:'file',accept:'image/png,image/jpeg,image/webp',multiple:max - value.length > 1,
              onChange:e => { add(e.target.files); e.target.value = '' }}),
            h('span',null,busy ? '送信中…' : '＋ 画像'))),
        hint && h('p',{className:'dsh-anima-meta'},hint),
        error && h('p',{className:'dsh-anima-error',role:'alert'},error))
    }

    function Inputs({paths}) {
      const tokens = paths.filter(Boolean).map(tokenOf)
      return tokens.length > 0 && h('div',{className:'dsh-anima-inputs'},
        tokens.map(token => h('img',{key:token,src:uploadUrl(token),alt:'入力画像',loading:'lazy'})))
    }

    const minutes = ms => ms < 60000 ? Math.max(1, Math.round(ms / 1000)) + ' 秒' : Math.round(ms / 60000) + ' 分'

    function GenerateJob({job, cancel}) {
      const r = job.request || {}
      const fraction = r.count ? ((job.done || 0) + (job.stage === 'generating' && job.step ? job.step / job.steps : 0)) / r.count : 0
      const images = job.images || []
      const each = images.length ? images.reduce((sum, image) => sum + image.seconds, 0) / images.length : 0
      return h('section',{className:'dsh-anima-card','aria-live':'polite'},
        h('header',null,h('strong',{className:'dsh-anima-name'},(r.kind === 'ph3' ? 'P-H3 画像 · ' + (MODE_LABEL[r.mode] ? MODE_LABEL[r.mode] + ' · ' : '') : '画像生成 · ') + r.count + ' 枚'),
          h('span',{className:'dsh-anima-stage dsh-anima-'+job.stage},
            job.queued ? STAGES.queued + '（' + job.position + ' 番目）' : STAGES[job.stage] || job.stage)),
        h('p',{className:'dsh-anima-prompt'},r.prompt),
        r.images && h(Inputs,{paths:r.images}),
        h('div',{className:'dsh-anima-meta'},(r.width ? r.width + '×' + r.height : '元画像の大きさ') + (r.kind === 'ph3' ? ' · 8 step' : ' · ' + r.steps + ' step · CFG ' + r.cfg + (r.fp8 === false ? ' · BF16' : ' · FP8')) + ' · seed ' + r.seed +
          (r.count > 1 ? '〜' + (r.seed + r.count - 1) : '') +
          (r.loras?.length ? ' · ' + r.loras.map(l => l.name.replace(/\.safetensors$/,'') + ' ×' + l.strength).join(', ') : '')),
        job.stage === 'generating' && h(React.Fragment,null,h('progress',{max:1,value:fraction}),
          h('div',{className:'dsh-anima-meta'},(job.done || 0) + ' / ' + r.count + ' 枚' + (job.step ? ' · step ' + job.step + ' / ' + job.steps : '') +
            (each ? ' · 1 枚 ' + each.toFixed(1) + ' 秒' : ''))),
        job.message && job.stage !== 'done' && h('p',{className:job.stage === 'error' ? 'dsh-anima-error' : 'dsh-anima-meta',role:job.stage === 'error' ? 'alert' : undefined},job.message),
        images.length > 0 && h('div',{className:'dsh-anima-images'},images.map(image =>
          h('a',{key:image.file,href:imageUrl(job, image.file),target:'_blank',rel:'noopener',title:'seed ' + image.seed},
            h('img',{src:imageUrl(job, image.file),alt:'seed ' + image.seed,loading:'lazy',style:r.width ? {aspectRatio:r.width + '/' + r.height} : undefined})))),
        job.stage === 'done' && h('p',{className:'dsh-anima-meta'},'読み込み ' + Math.round(job.load_seconds || 0) + ' 秒 · 1 枚 ' + each.toFixed(1) + ' 秒'),
        (job.running || job.queued) && h(Button,{variant:'outline',size:'sm',onClick:() => cancel(job.id)},job.running ? '中止' : '取り消す'))
    }

    function VideoJob({job, cancel}) {
      const r = job.request || {}
      const video = job.images?.[0]
      return h('section',{className:'dsh-anima-card','aria-live':'polite'},
        h('header',null,h('strong',{className:'dsh-anima-name'},'H3 動画' + (MODE_LABEL[r.mode] ? ' · ' + MODE_LABEL[r.mode] : '')),
          h('span',{className:'dsh-anima-stage dsh-anima-'+job.stage},
            job.queued ? STAGES.queued + '（' + job.position + ' 番目）' : STAGES[job.stage] || job.stage)),
        h('p',{className:'dsh-anima-prompt'},r.prompt),
        h(Inputs,{paths:[r.first_frame, r.last_frame, ...(r.references || [])]}),
        h('div',{className:'dsh-anima-meta'},r.width + '×' + r.height + ' · ' + r.length + ' フレーム · seed ' + r.seed),
        job.stage === 'generating' && h(React.Fragment,null,h('progress',{max:8,value:job.step || 0}),
          h('div',{className:'dsh-anima-meta'},'step ' + (job.step || 0) + ' / 8')),
        job.message && job.stage !== 'done' && h('p',{className:job.stage === 'error' ? 'dsh-anima-error' : 'dsh-anima-meta',role:job.stage === 'error' ? 'alert' : undefined},job.message),
        video && h('video',{className:'dsh-anima-video',src:imageUrl(job, video.file),controls:true,playsInline:true,preload:'metadata',
          style:{aspectRatio:r.width + '/' + r.height}}),
        job.stage === 'done' && h('p',{className:'dsh-anima-meta'},'読み込み ' + Math.round(job.load_seconds || 0) + ' 秒 · 生成 ' +
          Math.round(job.sample_seconds || 0) + ' 秒 · 合計 ' + Math.round(job.total_seconds || 0) + ' 秒'),
        (job.running || job.queued) && h(Button,{variant:'outline',size:'sm',onClick:() => cancel(job.id)},job.running ? '中止' : '取り消す'))
    }

    /** LoRA rows for the generation panels; `model` picks which LoRAs are listed (anima / h3 / ph3). */
    function LoraPicker({model, value, onChange, disabled, setError}) {
      const [available,setAvailable] = React.useState([])
      React.useEffect(() => { api({action:'loras',model}).then(data => setAvailable(data.loras)).catch(e => setError(e.message)) }, [model])
      const setLora = (index, patch) => onChange(value.map((item, i) => i === index ? {...item,...patch} : item))
      const label = {anima:'Anima',h3:'H3',ph3:'P-H3'}[model]
      return h('fieldset',{disabled},h('legend',null,'LoRA'),
        value.map((item, index) => h('div',{key:index,className:'dsh-anima-row'},
          h('select',{value:item.name,'aria-label':'LoRA',onChange:e => setLora(index,{name:e.target.value})},
            h('option',{value:''},'選択してください'),available.map(name => h('option',{key:name,value:name},name.replace(/\.safetensors$/,'')))),
          h('input',{type:'number',step:0.05,min:-2,max:2,value:item.strength,className:'dsh-anima-narrow','aria-label':'強さ',
            onChange:e => setLora(index,{strength:e.target.value})}),
          h(Button,{type:'button',size:'sm',variant:'outline',onClick:() => onChange(value.filter((_, i) => i !== index))},'外す'))),
        value.length < 4 && h(Button,{type:'button',size:'sm',variant:'outline',className:'dsh-anima-start',
          disabled:!available.length,onClick:() => onChange([...value,{name:'',strength:1}])},
          available.length ? 'LoRA を追加' : '使える ' + label + ' LoRA がありません'))
    }
    const chosen = loras => loras.filter(item => item.name).map(item => ({name:item.name,strength:Number(item.strength)}))

    function Ph3Panel({editing = false}) {
      const [form,setForm] = React.useState({prompt:'',mode:editing ? 'edit' : 't2i',images:[],size:editing ? 'auto' : '2912x2912',count:1,seed:'',loras:[]})
      const [jobs,setJobs] = useQueue()
      const [notice,setNotice] = React.useState('')
      const [error,setError] = React.useState('')
      const [busy,setBusy] = React.useState(false)
      const set = patch => setForm(previous => ({...previous,...patch}))
      async function submit(e) {
        e.preventDefault()
        setBusy(true); setError(''); setNotice('')
        try {
          const [width, height] = form.size === 'auto' ? [0, 0] : form.size.split('x').map(Number)
          const data = await api({action:'image',mode:form.mode,images:form.mode === 't2i' ? [] : form.images,prompt:form.prompt,
            width,height,count:Number(form.count),seed:form.seed === '' ? null : Number(form.seed),loras:chosen(form.loras)})
          setJobs(data.jobs)
          setNotice('キューに追加しました')
        } catch (e) { setError(e.message) }
        setBusy(false)
      }
      const active = jobs.filter(job => job.running || job.queued).length
      return h('section',{className:'dsh-anima-panel'},
        h('header',{className:'dsh-anima-head'},h('h1',null,h(ImageIcon),editing ? 'P-H3 画像編集' : 'P-H3 画像')),
        h('div',{className:'dsh-anima-body'},
          h('form',{className:'dsh-anima-card',onSubmit:submit},
            !editing && h('label',null,h('span',null,'モード'),h('select',{value:form.mode,disabled:busy,
              onChange:e => set({mode:e.target.value,size:e.target.value === 'edit' ? 'auto' : form.size === 'auto' ? '2912x2912' : form.size})},
              IMAGE_MODES.map(([value,label]) => h('option',{key:value,value},label)))),
            form.mode !== 't2i' && h(Pictures,{label:form.mode === 'edit' ? '画像（1 枚目を編集）' : '参照画像',max:9,value:form.images,disabled:busy,
              onChange:images => set({images}),
              captions:form.mode === 'edit' ? index => index ? '<Picture ' + (index + 1) + '>' : '編集する画像' : undefined,
              hint:form.mode === 'edit' ? 'プロンプトは編集の指示（例: 背景を夕焼けの海辺にする）。参照画像は <Picture 2> などで指せます。'
                : 'プロンプトで <Picture 1> などと参照画像を指してください（例: The girl from <Picture 1> reads a book in a park）。'}),
            h('label',null,h('span',null,form.mode === 'edit' ? '編集の指示' : 'プロンプト'),
              h('textarea',{value:form.prompt,rows:4,required:true,disabled:busy,
                placeholder:form.mode === 'edit' ? '例: Put the gold hair clip from <Picture 2> in her hair.'
                  : form.mode === 'reference' ? '例: The girl from <Picture 1>, wearing the hair clip from <Picture 2>, sits on a park bench.'
                  : '例: A neon sign that says "OPEN 24/7" above a ramen shop at night in the rain, cinematic photo',
                onChange:e => set({prompt:e.target.value}),onKeyDown:e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) e.currentTarget.form.requestSubmit() }})),
            h('div',{className:'dsh-anima-grid'},
              h('label',null,h('span',null,'サイズ'),h('select',{value:form.size,disabled:busy,onChange:e => set({size:e.target.value})},
                (form.mode === 'edit' ? [['auto','編集する画像の大きさ'], ...IMAGE_SIZES] : IMAGE_SIZES)
                  .map(([value,label]) => h('option',{key:value,value},label)))),
              h('label',null,h('span',null,'枚数'),h('input',{type:'number',min:1,max:16,value:form.count,disabled:busy,onChange:e => set({count:e.target.value})})),
              h('label',null,h('span',null,'シード'),h('input',{type:'number',min:0,value:form.seed,placeholder:'ランダム',disabled:busy,onChange:e => set({seed:e.target.value})}))),
            h(LoraPicker,{model:'ph3',value:form.loras,onChange:loras => set({loras}),disabled:busy,setError}),
            h('div',{className:'dsh-anima-actions'},
              h(Button,{type:'submit',variant:'primary',disabled:busy || !form.prompt.trim() || (form.mode !== 't2i' && !form.images.length)},
                active ? 'キューに追加（実行中・待機中 ' + active + ' 件）' : editing ? '編集' : '生成')),
            h('p',{className:'dsh-anima-meta'},'p-h3-image int8 · euler 8 step。プロンプトはそのまま使います（プロンプト拡張なし）。'),
            notice && h('p',{className:'dsh-anima-meta',role:'status'},notice),
            error && h('p',{className:'dsh-anima-error',role:'alert'},error)),
          h(Queue,{jobs,setJobs,setError})))
    }

    function Job({job, cancel}) {
      if (job.request?.kind === 'generate') return h(GenerateJob,{job,cancel})
      if (job.request?.kind === 'h3') return h(VideoJob,{job,cancel})
      if (job.request?.kind === 'ph3') return h(GenerateJob,{job,cancel})
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

    function ImageIcon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('rect',{x:3.5,y:3.5,width:17,height:17,rx:2.5}),h('circle',{cx:9,cy:9,r:1.8}),h('path',{d:'M3.5 17l5-5 4 4 3-3 5 5'}))
    }

    function registerPanel(ctx, id, order, label, Panel, Icon) {
      ctx.slots.inject('main', () => ctx.slots.register({name:'main',key:id}, Panel))
      ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({name:'sidebar.panellist',id,order,label:() => label}, Icon))
    }

    return {React,h,Button,api,Pictures,LoraPicker,chosen,useQueue,Queue,Ph3Panel,ImageIcon,PURPOSES,LOCAL,toBase64,registerPanel,
      inject:['slots'],apply(ctx) {
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
        .dsh-anima-pictures{display:grid;gap:6px}
        .dsh-anima-thumbs{display:flex;flex-wrap:wrap;gap:8px}
        .dsh-anima-thumbs figure{position:relative;margin:0;width:88px;display:grid;gap:2px}
        .dsh-anima-thumbs img{width:88px;height:88px;object-fit:cover;border-radius:var(--dsw-radius-md);border:1px solid var(--dsw-alias-border-l3);background:var(--dsw-alias-bg-base)}
        .dsh-anima-thumbs figcaption{font-size:11px;color:var(--dsw-alias-label-secondary);text-align:center;overflow-wrap:anywhere}
        .dsh-anima-thumbs button{position:absolute;top:2px;right:2px;width:24px;height:24px;border:0;border-radius:999px;background:rgba(0,0,0,.6);color:#fff;cursor:pointer;font:inherit;line-height:1}
        .dsh-anima-add{position:relative;display:flex!important;align-items:center;justify-content:center;min-width:88px;min-height:88px;padding:8px;box-sizing:border-box;border:1px dashed var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);cursor:pointer;background:var(--dsw-alias-bg-base);text-align:center}
        .dsh-anima-add input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%}
        .dsh-anima-add span{font-size:13px!important;color:inherit!important;font-weight:400!important}
        .dsh-anima-upload{min-height:40px}
        .dsh-anima-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
        .dsh-anima-inputs{display:flex;flex-wrap:wrap;gap:6px}
        .dsh-anima-inputs img{width:48px;height:48px;object-fit:cover;border-radius:6px;border:1px solid var(--dsw-alias-border-l3)}
        .dsh-anima-video{display:block;width:100%;height:auto;border-radius:var(--dsw-radius-md);background:#000}
        @media(max-width:650px){.dsh-anima-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.dsh-anima-grid label:first-child{grid-column:1/-1}}
        @media(max-width:650px){.dsh-anima-purposes{grid-template-columns:1fr}.dsh-anima-head{padding:8px 12px}.dsh-anima-body{padding:16px 12px}}
      `
      document.head.append(style)
      ctx.effect(() => () => style.remove(), 'Anima LoRA panel styles')
    }}
  }
})
