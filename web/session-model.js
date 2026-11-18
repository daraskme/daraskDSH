// Supply the existing composer menu with its host catalog before a Session is available.
function studioModelSnapshot(snapshot, selection) {
  return {
    current:selection ?? snapshot.value?.default ?? null,
    groups:snapshot.value?.groups ?? [],
    failures:snapshot.value?.failures ?? [],
    status:snapshot.status, error:snapshot.error, pending:null
  }
}
function studioDraftActions(models, ctx) {
  return {
    available:true, isDraft:true, directory:models.catalog.store,
    load:() => models.catalog.load().catch(() => {}),
    resolveText:text => ctx.locale.resolveText(text),
    select:async selection => {
      models.studioDraft = selection
      return {ok:true}
    }
  }
}

function ModelSelect({locked,studioModel,isDraft,available,directory,load,select,resolveText}) {
  const snapshot = react.useSyncExternalStore(fn=>directory.subscribe(fn),()=>directory.getSnapshot())
  const state = isDraft ? studioModelSnapshot(snapshot,studioModel?.draft) : snapshot
  const [open,setOpen] = react.useState(false), [data,setData] = react.useState(null)
  const [service,setService] = react.useState('codex'), [value,setValue] = react.useState({})
  const [busy,setBusy] = react.useState(false), [notice,setNotice] = react.useState('')
  const {Button,Modal,IconChevronDownOutlineRegular,IconRefreshOutlineRegular} = _deepseek_ai_dsh_client_ui_primitives
  const groups = state.groups.filter(g=>g.id!=='local-cli')
  const nativeGroup = groups.find(g=>g.id===service)
  const nativeModel = nativeGroup?.models.find(m=>m.id===value.model)
  const nativeCap = nativeGroup && {models:nativeGroup.models.map(m=>m.id),variants:nativeGroup.models.map(m=>({id:m.id,label:resolveText(m.name)})),byModel:Object.fromEntries(nativeGroup.models.map(m=>[m.id,{efforts:m.reasoning?.efforts.map(e=>e.id) || []}])),efforts:nativeModel?.reasoning?.efforts.map(e=>e.id) || []}
  const cap = data?.catalog[service] || nativeCap
  const current = smDecode(state.current)
  const currentOptions = current?.options
  const currentName = current ? [smServices[current.service] || current.service,currentOptions?.model,currentOptions?.effort,currentOptions?.fast && 'Fast',currentOptions?.ultracode && 'Ultracode'].filter(Boolean).join(' · ')
    : state.current ? [groups.find(g=>g.id===state.current.provider)?.name || state.current.provider,state.current.model,state.current.reasoningEffort].filter(Boolean).map(resolveText).join(' · ') : '未選択'
  react.useEffect(()=>{load()},[])
  react.useEffect(()=>{
    if(!open) return
    const ctrl = new AbortController()
    setData(null);setNotice('読み込み中…')
    smFetchOptions(false,ctrl.signal).then(d=>{
      if(ctrl.signal.aborted) return
      setData(d);setNotice('')
      if(current) {setService(current.service);setValue(current.options ?? {...d.values[current.service]})}
      else if(state.current) {setService(state.current.provider);setValue({model:state.current.model,...state.current.reasoningEffort && {effort:state.current.reasoningEffort}})}
      else {setService('codex');setValue({...d.values.codex})}
    }).catch(e=>{if(!ctrl.signal.aborted)setNotice(e.message)})
    return ()=>ctrl.abort()
  },[open])
  function close() {if(!busy) setOpen(false)}
  async function refresh() {
    setBusy(true);setNotice('一覧を更新中…')
    try{setData(await smFetchOptions(true));setNotice('一覧を更新しました')}
    catch(e){setNotice(e.message)}finally{setBusy(false)}
  }
  async function apply() {
    setBusy(true);setNotice('');studioModel?.onBusy(true)
    try {
      const selection = data?.catalog[service] ? smEncode(service,value) : {provider:service,model:value.model,...value.effort && {reasoningEffort:value.effort}}
      if(!selection.model) throw Error('モデルを選択してください')
      const result = await select(selection)
      if(!result?.ok) throw Error(result?.error?.message || 'モデルを変更できません')
      studioModel?.onSelect(selection);setOpen(false)
    } catch(e) {setNotice(e.message)}
    finally{setBusy(false);studioModel?.onBusy(false)}
  }
  return sm('div',{className:'ds-model-select',onClick:e=>e.stopPropagation()},
    sm(Button,{type:'button',size:'sm',className:'ds-model-trigger',disabled:locked || !available || busy,
      'aria-label':'モデルを選択、現在 '+currentName,title:currentName,'aria-haspopup':'dialog','aria-expanded':open,onClick:()=>setOpen(true)},
      sm('span',null,currentName),sm(IconChevronDownOutlineRegular,{size:12})),
    open && sm(Modal,{open:true,title:'LLMサービス・モデル選択',closeLabel:'閉じる',onClose:close,className:'ds-model-dialog',
      footer:sm(react.Fragment,null,
        sm(Button,{type:'button',variant:'outline',disabled:busy,onClick:close},'キャンセル'),
        sm(Button,{type:'button',variant:'primary',disabled:busy || !data || !cap,onClick:apply},'適用'))},
      sm('div',{className:'ds-options ds-model-body'},
      sm('label',null,'使用するLLMサービス',sm('select',{'aria-label':'使用するLLMサービス',value:service,disabled:busy || !data,onChange:e=>{
        const next=e.target.value;setService(next);setNotice('')
        setValue(data.catalog[next] ? {...data.values[next]} : {model:groups.find(g=>g.id===next)?.models[0]?.id || ''})
      }},Object.entries(data?.services || smServices).map(([id,name])=>sm('option',{key:id,value:id},name)),
      groups.map(g=>sm('option',{key:g.id,value:g.id},resolveText(g.name))))),
      data && cap && sm(ServiceOptions,{service,cap,value,disabled:busy,onChange:setValue,defaultLabel:nativeGroup?'モデルを選択':'サービスの既定値'}),
      notice && sm('div',{role:'status'},notice),
      sm(Button,{type:'button',size:'sm',className:'ds-refresh',icon:sm(IconRefreshOutlineRegular,{size:14}),disabled:busy,onClick:refresh},'一覧を更新'))))
}
