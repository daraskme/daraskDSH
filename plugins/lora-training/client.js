window.__ModuleLoader__.load({
  id:'dsh-anima-lora-trainer',
  factory(require) {
    const {React,h,Button,api,PURPOSES,LOCAL,toBase64,useQueue,Queue,registerPanel} = require('dsh-media-shared')

    const {IconFolderOpenRegular} = require('@deepseek-ai/dsh-client-ui-primitives')
    const MODELS = [['anima','Anima','イラスト（画像）'],['h3','MiniMax H3','動画（音声つき）・画像'],['ph3','P-H3','写真・一般の画像']]

    const lr = value => value < 0.001 ? value.toExponential().replace('e-', 'e-') : String(value)
    function Settings({plan}) {
      const s = plan.settings, a = s.args
      const h3 = s.model && s.model !== 'anima'
      const rows = [
        [h3 ? '素材' : '画像', plan.images + (h3 ? ' 件' : ' 枚') + '（キャプションあり ' + plan.captioned + '）'],
        ['学習量', s.epochs + ' epoch · ' + s.steps + ' step'],
        ['解像度 / rank', (s.model === 'h3' ? '画像 768px・動画 512px' : s.resolution + 'px') + ' · rank ' + s.rank],
        ['学習率', lr(a['learning-rate']) + (a['cross-attn-lr'] !== undefined ? '（cross-attn ' + lr(a['cross-attn-lr']) + '）' : '')],
        h3 ? ['スケジュール', 'warmup 10 step · 一定 · CFG 4 で蒸留を保つ']
           : ['スケジュール', 'warmup ' + Math.round(a.warmup * 100) + '% · cosine ' + a['lr-cycles'] + ' 回 · timestep ×' + a['timestep-scale']],
        ['出力', plan.comfyName],
      ]
      return h('dl',{className:'dsh-anima-settings'},rows.flatMap(([k,v])=>[h('dt',{key:k},k),h('dd',{key:k+'v'},v)]))
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
          LOCAL && h(Button,{type:'button',size:'sm',variant:'outline',onClick:desktop},'Ubuntu で選択'),
          h(Button,{type:'button',size:'sm',variant:'outline',onClick:close},'閉じる')),
        view && h('ul',null,
          view.parent && h('li',null,h('button',{type:'button',onClick:() => open(view.parent)},'↑ 上のフォルダ')),
          view.dirs.map(dir => h('li',{key:dir.path},h('button',{type:'button',onClick:() => open(dir.path)},
            h(IconFolderOpenRegular,{size:14}),h('span',null,dir.name),dir.images > 0 && h('small',null,dir.images + ' 枚'))))),
        view && h(Button,{type:'button',variant:'primary',size:'sm',disabled:!view.images,onClick:() => choose(view.path)},
          view.images ? 'このフォルダを使う（画像 ' + view.images + ' 枚）' : 'このフォルダに画像はありません'),
        error && h('p',{className:'dsh-anima-error',role:'alert'},error))
    }

    /** Sends a training set from this device (images and same-stem .txt captions, one file per request) into a new
     *  folder on the server; existing files are never overwritten. */
    function DatasetUpload({disabled, done}) {
      const [progress,setProgress] = React.useState(null)
      const [error,setError] = React.useState('')
      async function send(files) {
        files = [...files]
        if (!files.length) return
        const dataset = 'upload-' + new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')
        setError(''); setProgress({sent:0,total:files.length})
        try {
          let folder
          for (const [index, file] of files.entries()) {
            if (file.size > 20 << 20) throw Error(file.name + ' は 20 MB を超えています')
            ;({folder} = await api({action:'dataset',dataset,name:file.name,data:await toBase64(file)}))
            setProgress({sent:index + 1,total:files.length})
          }
          done(folder, files.some(file => file.name.toLowerCase().endsWith('.txt')))
        } catch (e) { setError(e.message) }
        setProgress(null)
      }
      return h(React.Fragment,null,
        h('label',{className:'dsh-anima-add dsh-anima-upload'},
          h('input',{type:'file',multiple:true,accept:'image/png,image/jpeg,image/webp,.txt',disabled:disabled || !!progress,
            onChange:e => { send(e.target.files); e.target.value = '' }}),
          h('span',null,progress ? '送信中 ' + progress.sent + ' / ' + progress.total : 'この端末の画像（と同じ名前の .txt）を送って使う')),
        error && h('p',{className:'dsh-anima-error',role:'alert'},error))
    }

    function Panel() {
      const [form,setForm] = React.useState({model:'anima',folder:'',purpose:'style',trigger:'',name:'',overwrite:true})
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
        h('header',{className:'dsh-anima-head'},h('h1',null,h(Icon),'LoRA 学習')),
        h('div',{className:'dsh-anima-body'},
          h('form',{className:'dsh-anima-card',onSubmit:e => { e.preventDefault(); run('plan') }},
            h('fieldset',{disabled},h('legend',null,'モデル'),
              h('div',{className:'dsh-anima-purposes'},MODELS.map(([key,label,hint]) =>
                h('label',{key,className:form.model === key ? 'is-selected' : ''},
                  h('input',{type:'radio',name:'model',value:key,checked:form.model === key,onChange:() => set({model:key})}),
                  h('strong',null,label),h('small',null,hint))))),
            h('label',null,h('span',null,form.model === 'h3' ? '画像・動画フォルダ（動画は先頭の最大 136 フレームと音声）' : '画像フォルダ'),
              h('div',{className:'dsh-anima-row'},
                h('input',{value:form.folder,placeholder:'/path/to/images',disabled,required:true,onChange:e => set({folder:e.target.value})}),
                h(Button,{type:'button',variant:'outline',disabled,className:'dsh-anima-nowrap',icon:h(IconFolderOpenRegular,{size:15}),onClick:() => setPicking(!picking)},'選択…')),
              picking && h(FolderBrowser,{start:form.folder.trim(),close:() => setPicking(false),
                choose:path => { set({folder:path}); setPicking(false) }}),
              h(DatasetUpload,{disabled,done:(folder, captioned) => set({folder,overwrite:!captioned})})),
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

    return {inject:['slots'],apply(ctx) {
      registerPanel(ctx, 'anima-lora', 21, 'LoRA 学習', Panel, Icon)
    }}
  }
})
