window.__ModuleLoader__.load({
  id:'dsh-anima-lora-trainer',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const {Button, IconFolderOpenRegular} = require('@deepseek-ai/dsh-client-ui-primitives')
    const API = '/api/darask.animaLora'
    const PURPOSES = [['character','キャラ','一人の人物・キャラの見た目を覚えさせる'],
      ['style','スタイル','絵柄・画風を覚えさせる'],['concept','コンセプト','ポーズ・物・状況などの概念を覚えさせる']]
    const STAGES = {starting:'準備中',captioning:'キャプション作成',training:'学習',done:'完了',error:'エラー',cancelled:'中止'}

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

    function Progress({job, cancel}) {
      const fraction = job.total ? job.done / job.total : 0
      const eta = job.stage === 'training' && job.step_ms && job.total ? (job.total - job.done) * job.step_ms : 0
      return h('section',{className:'dsh-anima-card','aria-live':'polite'},
        h('header',null,h('strong',null,'学習ジョブ'),h('span',{className:'dsh-anima-stage dsh-anima-'+job.stage},STAGES[job.stage] || job.stage)),
        h('div',{className:'dsh-anima-meta'},job.request?.comfyName),
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
        job.running && h(Button,{variant:'outline',onClick:cancel},'中止'))
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

    function Panel() {
      const [form,setForm] = React.useState({folder:'',purpose:'style',trigger:'',name:'',overwrite:true})
      const [plan,setPlan] = React.useState(null)
      const [job,setJob] = React.useState(null)
      const [error,setError] = React.useState('')
      const [busy,setBusy] = React.useState(false)
      const [picking,setPicking] = React.useState(false)
      const running = job?.running
      const set = patch => { setForm(previous => ({...previous,...patch})); setPlan(null) }

      React.useEffect(() => {
        let stop = false, timer
        const poll = async () => {
          try { const data = await api(); if (!stop) setJob(data.job) } catch {}
          if (!stop) timer = setTimeout(poll, 2000)
        }
        poll()
        return () => { stop = true; clearTimeout(timer) }
      }, [])

      async function run(action) {
        setBusy(true); setError('')
        try {
          const data = await api({action,...form})
          if (action === 'plan') setPlan(data)
          else { setJob(data.job); setPlan(null) }
        } catch (e) { setError(e.message) }
        setBusy(false)
      }
      const cancel = () => api({action:'cancel'}).then(data => setJob(data.job)).catch(e => setError(e.message))

      const disabled = busy || running
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
                'キャプション作成と学習を開始')),
            plan && h(Settings,{plan}),
            plan?.exists && h('p',{className:'dsh-anima-error',role:'alert'},'同じ名前の LoRA があります。出力名を変えてください。'),
            error && h('p',{className:'dsh-anima-error',role:'alert'},error)),
          job && h(Progress,{job,cancel})))
    }

    function Icon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('rect',{x:3.5,y:4.5,width:17,height:15,rx:2.5}),
        h('path',{d:'M3.5 15.5l4.5-4.5 3.5 3.5 2.5-2.5 6.5 6.5'}),
        h('path',{d:'M16.5 6.8l.6 1.6 1.6.6-1.6.6-.6 1.6-.6-1.6-1.6-.6 1.6-.6z',fill:'currentColor',stroke:'none'}))
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
        .dsh-anima-card input:not([type]),.dsh-anima-card input[type=text]{box-sizing:border-box;width:100%;min-width:0;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);padding:8px 10px;font:inherit;color:inherit;background:var(--dsw-alias-bg-base)}
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
        .dsh-anima-card header{display:flex;align-items:center;justify-content:space-between;gap:8px}
        .dsh-anima-stage{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--dsw-alias-border-l3)}
        .dsh-anima-done{color:var(--dsw-alias-state-success-primary,inherit)}
        .dsh-anima-error{color:var(--dsw-alias-state-error-primary);margin:0;overflow-wrap:anywhere}
        .dsh-anima-meta{font-size:13px;color:var(--dsw-alias-label-secondary);margin:0;overflow-wrap:anywhere}
        .dsh-anima-card progress{width:100%;height:8px}
        .dsh-anima-card blockquote{margin:0;padding:8px 10px;border-left:3px solid var(--dsw-alias-border-l3);font-size:12px;color:var(--dsw-alias-label-secondary);overflow-wrap:anywhere}
        .dsh-anima-card p{margin:0}
        @media(max-width:650px){.dsh-anima-purposes{grid-template-columns:1fr}.dsh-anima-head{padding:8px 12px}.dsh-anima-body{padding:16px 12px}}
      `
      document.head.append(style)
      ctx.effect(() => () => style.remove(), 'Anima LoRA panel styles')
      ctx.slots.inject('main', () => ctx.slots.register({name:'main',key:'anima-lora'}, Panel))
      ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
        name:'sidebar.panellist',id:'anima-lora',order:21,label:() => 'Anima LoRA'
      }, Icon))
    }}
  }
})
