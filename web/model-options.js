// Shared fields embedded into both the settings and composer bundles.
const sm = react.createElement
const smServices = {auto:'自動引き継ぎ（Cursor → Astra → Opus）',google:'Google AI Ultra','devin-free':'Devin（無料のみ）',devin:'Devin',muse:'Muse',grok:'Grok',cursor:'Cursor',codex:'Codex',claude:'Claude Code'}
async function smFetchOptions(refresh = false, signal) {
  const response = await fetch('/api/darask.options'+(refresh?'?refresh=1':''),{signal})
  const data = await response.json()
  if (!response.ok) throw Error(data.error || 'モデル一覧を取得できません')
  return data
}
function smDecode(selection) {
  if (selection?.provider !== 'local-cli') return null
  const [service,encoded] = selection.model.split('@')
  try { return {service,options:encoded ? JSON.parse(atob(encoded.replace(/-/g,'+').replace(/_/g,'/'))) : undefined} }
  catch { return null }
}
function smEncode(service,options) {
  if (options.model !== undefined && !/^[a-zA-Z0-9][a-zA-Z0-9_.,:/=[\]-]{0,180}$/.test(options.model)) throw Error('モデル ID に使用できない文字が含まれています')
  return {provider:'local-cli',model:service+'@'+btoa(JSON.stringify(options)).replace(/=/g,'').replace(/\+/g,'-').replace(/\//g,'_')}
}
function ServiceOptions({service,cap,value,onChange,disabled,defaultLabel='サービスの既定値'}) {
  const [manual,setManual] = react.useState(false)
  react.useEffect(()=>setManual(false),[service])
  const models = cap.models || [], specific = cap.byModel?.[value.model] || cap
  const set = (key,next)=>{
    const row = {...value}
    if (next === '') delete row[key]
    else row[key] = next
    if (key === 'model') {
      const nextCap = cap.byModel?.[next] || cap
      if (!(nextCap.efforts || []).includes(row.effort)) delete row.effort
      if (!nextCap.fast) delete row.fast
      if (!(nextCap.ultracode ?? cap.ultracode)) delete row.ultracode
    }
    onChange(row)
  }
  const label = id => cap.variants?.find(v=>v.id===id)?.label || id
  if(service==='auto')return sm('div',null,sm('p',null,'Cursor → Codex Astra 6.2 → Claude Opus 5.5'),sm('small',null,'一般チャット用の自動引き継ぎです。TWAはサブスク枠のAstra 6.2が統括し、Cursor・Grokが通常の並列作業を担当します。Cursor/Grokの未解決分はAstraへ、Astraでも解けないものはOpusへ引き継ぎます。'))
  return sm(react.Fragment,null,
    sm('label',null,'モデルの種類',sm('select',{'aria-label':'モデルの種類',disabled,value:manual?'__custom':value.model || '',onChange:e=>{
      if(e.target.value==='__custom') setManual(true)
      else {setManual(false);set('model',e.target.value)}
    }},sm('option',{value:''},defaultLabel),
    value.model && !models.includes(value.model) && sm('option',{value:value.model},value.model),
    models.map(id=>sm('option',{key:id,value:id},label(id))),sm('option',{value:'__custom'},'モデル ID を入力…'))),
    manual && sm('label',null,'モデル ID',sm('input',{value:value.model || '',disabled,placeholder:'モデル ID',onChange:e=>set('model',e.target.value.trim())})),
    service==='google' && sm('small',null,cap.unavailable ? 'Google に接続できません。Ubuntu の端末で dsh login google を実行し、一覧を更新してください。' : 'Google AI Ultra の契約枠で Antigravity を利用します。'),
    service==='cursor' && sm('small',null,'エフォートと高速設定はモデル名で選択します。'),
    service==='grok' && sm('small',null,'高速設定は Fast モデルで選択します。'),
    service==='codex' && value.subscription && sm('small',null,'ChatGPTログインのサブスク枠を使用します。APIキーへ自動で切り替えません。'),
    (service==='devin' || service==='devin-free') && sm('small',null,cap.unavailable?'Devin にログインして一覧を更新してください。':(cap.variants?.find(v=>v.id===value.model)?.price || '未指定の場合は無料モデルを自動選択します。')),
    sm('fieldset',{className:'ds-model-options',disabled},sm('legend',null,'エフォート・オプション'),
      specific.efforts?.length>0 ? sm('label',null,'エフォート',sm('select',{'aria-label':'エフォート',value:value.effort || '',onChange:e=>set('effort',e.target.value)},
        ['',...specific.efforts].map(id=>sm('option',{key:id,value:id},id || defaultLabel)))) : sm('small',null,'このモデルの追加エフォート設定はありません。'),
      ['fast','ultracode'].filter(k=>specific[k] ?? cap[k]).map(k=>sm('label',{className:'ds-check',key:k},sm('input',{type:'checkbox',checked:!!value[k],onChange:e=>set(k,e.target.checked)}),k==='fast'?'高速モード':'Ultracode'))))
}
