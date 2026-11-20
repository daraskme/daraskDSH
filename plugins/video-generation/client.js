window.__ModuleLoader__.load({
  id:'dsh-video-generation',
  factory(require) {
    const {React,h,Button,api,Pictures,LoraPicker,chosen,useQueue,Queue,registerPanel} = require('dsh-media-shared')

    const VIDEO_SIZES = [['1280x704','横 1280×704'],['704x1280','縦 704×1280'],['960x960','正方形 960×960'],['640x384','横 640×384（試し用）']]
    const LENGTHS = [[39,'1.6 秒（39 フレーム）'],[73,'3.0 秒（73 フレーム）'],[124,'5.2 秒（124 フレーム）'],[175,'7.3 秒（175 フレーム）'],[362,'15 秒（362 フレーム）'],[702,'29 秒（長尺 2 区間）'],[1042,'43 秒（長尺 3 区間）'],[1382,'58 秒（長尺 4 区間）']]

    const VIDEO_MODES = [['t2v','テキストから'],['i2v','画像から（最初・最後のフレーム）'],['ref','参照画像から']]

    function VideoPanel() {
      const [form,setForm] = React.useState({prompt:'',mode:'t2v',first:[],last:[],references:[],size:'1280x704',length:124,seed:'',loras:[]})
      const [jobs,setJobs] = useQueue()
      const [notice,setNotice] = React.useState('')
      const [error,setError] = React.useState('')
      const [busy,setBusy] = React.useState(false)
      const set = patch => setForm(previous => ({...previous,...patch}))
      async function submit(e) {
        e.preventDefault()
        setBusy(true); setError(''); setNotice('')
        try {
          const [width, height] = form.size.split('x').map(Number)
          const pictures = form.mode === 'i2v' ? {first:form.first[0] || null,last:form.last[0] || null}
            : form.mode === 'ref' ? {references:form.references} : {}
          const data = await api({action:'video',mode:form.mode,...pictures,prompt:form.prompt,width,height,length:Number(form.length),
            seed:form.seed === '' ? null : Number(form.seed),loras:chosen(form.loras)})
          setJobs(data.jobs)
          setNotice('キューに追加しました')
        } catch (e) { setError(e.message) }
        setBusy(false)
      }
      const active = jobs.filter(job => job.running || job.queued).length
      return h('section',{className:'dsh-anima-panel'},
        h('header',{className:'dsh-anima-head'},h('h1',null,h(VideoIcon),'H3 動画')),
        h('div',{className:'dsh-anima-body'},
          h('form',{className:'dsh-anima-card',onSubmit:submit},
            h('label',null,h('span',null,'モード'),h('select',{value:form.mode,disabled:busy,onChange:e => set({mode:e.target.value})},
              VIDEO_MODES.map(([value,label]) => h('option',{key:value,value},label)))),
            form.mode === 'i2v' && h('div',{className:'dsh-anima-pair'},
              h(Pictures,{label:'最初のフレーム',max:1,value:form.first,disabled:busy,onChange:first => set({first})}),
              h(Pictures,{label:'最後のフレーム（任意）',max:1,value:form.last,disabled:busy,onChange:last => set({last})})),
            form.mode === 'i2v' && h('p',{className:'dsh-anima-meta'},'最初のフレームは動画の大きさに引き伸ばし、最後のフレームは中央を切り抜きます。'),
            form.mode === 'ref' && h(Pictures,{label:'参照画像',max:9,value:form.references,disabled:busy,onChange:references => set({references}),
              hint:'プロンプトで <Picture 1> などと指してください（例: <Picture 1> waves at the camera in a sunny park）。'}),
            h('label',null,h('span',null,'プロンプト（映像と音の説明）'),
              h('textarea',{value:form.prompt,rows:4,required:true,disabled:busy,
                placeholder:'例: A close-up of a blue ceramic cup on a wooden table by a window. The camera moves slowly from left to right. Soft morning light, quiet room tone.',
                onChange:e => set({prompt:e.target.value}),onKeyDown:e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) e.currentTarget.form.requestSubmit() }})),
            h('div',{className:'dsh-anima-grid'},
              h('label',null,h('span',null,'サイズ'),h('select',{value:form.size,disabled:busy,onChange:e => set({size:e.target.value})},
                VIDEO_SIZES.map(([value,label]) => h('option',{key:value,value},label)))),
              h('label',null,h('span',null,'長さ'),h('select',{value:form.length,disabled:busy,onChange:e => set({length:e.target.value})},
                LENGTHS.map(([value,label]) => h('option',{key:value,value},label)))),
              h('label',null,h('span',null,'シード'),h('input',{type:'number',min:0,value:form.seed,placeholder:'ランダム',disabled:busy,onChange:e => set({seed:e.target.value})}))),
            h(LoraPicker,{model:'h3',value:form.loras,onChange:loras => set({loras}),disabled:busy,setError}),
            h('div',{className:'dsh-anima-actions'},
              h(Button,{type:'submit',variant:'primary',disabled:busy || !form.prompt.trim() ||
                (form.mode === 'i2v' && !form.first.length && !form.last.length) || (form.mode === 'ref' && !form.references.length)},
                active ? 'キューに追加（実行中・待機中 ' + active + ' 件）' : '生成')),
            h('p',{className:'dsh-anima-meta'},'Eros beta7 + VBVR Pro V2 · res_multistep 8 step · SLA。音声つき MP4（24 fps）。'),
            notice && h('p',{className:'dsh-anima-meta',role:'status'},notice),
            error && h('p',{className:'dsh-anima-error',role:'alert'},error)),
          h(Queue,{jobs,setJobs,setError})))
    }

    function VideoIcon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('rect',{x:3,y:6,width:13,height:12,rx:2.5}),h('path',{d:'M16 10.5l5-3v9l-5-3z'}))
    }

    return {inject:['slots'],apply(ctx) {
      registerPanel(ctx, 'h3-video', 23, 'H3 動画', VideoPanel, VideoIcon)
    }}
  }
})
