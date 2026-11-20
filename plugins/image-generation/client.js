window.__ModuleLoader__.load({
  id:'dsh-image-generation',
  factory(require) {
    const {React,h,Button,api,LoraPicker,chosen,useQueue,Queue,Ph3Panel,ImageIcon,registerPanel} = require('dsh-media-shared')

    const SIZES = [['832x1216','縦 832×1216'],['1216x832','横 1216×832'],['1024x1024','正方形 1024×1024'],
      ['896x1152','縦 896×1152'],['1152x896','横 1152×896']]

    function GeneratePanel() {
      const [form,setForm] = React.useState({prompt:'',negative:'',size:'832x1216',steps:34,cfg:4.5,seed:'',count:1,loras:[],exact:false})
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
          const data = await api({action:'generate',prompt:form.prompt,negative:form.negative,width,height,steps:Number(form.steps),
            cfg:Number(form.cfg),count:Number(form.count),seed:form.seed === '' ? null : Number(form.seed),
            loras:chosen(form.loras),fp8:!form.exact})
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
            h(LoraPicker,{model:'anima',value:form.loras,onChange:loras => set({loras}),disabled:busy,setError}),
            h('div',{className:'dsh-anima-grid'},
              h('label',null,h('span',null,'サイズ'),h('select',{value:form.size,disabled:busy,onChange:e => set({size:e.target.value})},
                SIZES.map(([value,label]) => h('option',{key:value,value},label)))),
              h('label',null,h('span',null,'枚数'),h('input',{type:'number',min:1,max:16,value:form.count,disabled:busy,onChange:e => set({count:e.target.value})})),
              h('label',null,h('span',null,'シード'),h('input',{type:'number',min:0,value:form.seed,placeholder:'ランダム',disabled:busy,onChange:e => set({seed:e.target.value})})),
              h('label',null,h('span',null,'step'),h('input',{type:'number',min:1,max:100,value:form.steps,disabled:busy,onChange:e => set({steps:e.target.value})})),
              h('label',null,h('span',null,'CFG'),h('input',{type:'number',min:1,max:20,step:0.1,value:form.cfg,disabled:busy,onChange:e => set({cfg:e.target.value})}))),
            h('label',{className:'dsh-anima-check'},
              h('input',{type:'checkbox',checked:form.exact,disabled:busy,onChange:e => set({exact:e.target.checked})}),
              h('span',null,'ComfyUI と同じ画像にする（BF16。約 1.4 倍遅い）')),
            h('div',{className:'dsh-anima-actions'},
              h(Button,{type:'submit',variant:'primary',disabled:busy || !form.prompt.trim()},
                active ? 'キューに追加（実行中・待機中 ' + active + ' 件）' : '生成')),
            h('p',{className:'dsh-anima-meta'},'er_sde · simple · Anima base v1.0。既定は FP8 で高速（構図は同じで細部が変わります）。'),
            notice && h('p',{className:'dsh-anima-meta',role:'status'},notice),
            error && h('p',{className:'dsh-anima-error',role:'alert'},error)),
          h(Queue,{jobs,setJobs,setError})))
    }

    function GenerateIcon() {
      return h('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        h('path',{d:'M4 20l9.5-9.5'}),h('path',{d:'M12 9l3 3'}),
        h('path',{d:'M16.5 3.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z',fill:'currentColor',stroke:'none'}),
        h('path',{d:'M19.5 11.5l.5 1.2 1.2.5-1.2.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z',fill:'currentColor',stroke:'none'}),
        h('path',{d:'M8.5 3.5l.5 1.2 1.2.5-1.2.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z',fill:'currentColor',stroke:'none'}))
    }

    return {inject:['slots'],apply(ctx) {
      registerPanel(ctx, 'anima-generate', 22, 'Anima 生成', GeneratePanel, GenerateIcon)
      registerPanel(ctx, 'ph3-image', 24, 'P-H3 画像生成', Ph3Panel, ImageIcon)
    }}
  }
})
