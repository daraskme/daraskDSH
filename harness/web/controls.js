// This module is embedded in the pinned settings client bundle by scripts/patch-ui.mjs.
const dsElement = react.createElement
const dsUI = _deepseek_ai_dsh_client_ui_primitives
const dsAgents = {muse:'Muse',grok:'Grok',cursor:'Cursor',codex:'Codex',claude:'Claude'}
const dsModes = {light:[dsUI.IconLightOutlineMedium,'ライトモード'],dark:[dsUI.IconDarkOutlineMedium,'ダークモード'],paper:[dsUI.IconDeliverDocRegular,'ペーパーモード']}
const dsDate = new Intl.DateTimeFormat('ja-JP',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'Asia/Tokyo'})
async function dsFetch(path, init) {
  const response = await fetch('/api/darask.'+path,init)
  if (!response.ok) throw Error((await response.json().catch(()=>({}))).error || '取得できませんでした')
  return response.json()
}
function DaraskFooter({wide, openSettings, theme}) {
  const [usage,setUsage] = react.useState({})
  const [mode,setMode] = react.useState(()=>localStorage.getItem('darask-mode') || theme()?.getTheme()?.active?.id || 'light')
  react.useEffect(()=>{
    const controller = new AbortController()
    const refresh = ()=>dsFetch('usage',{signal:controller.signal}).then(setUsage).catch(()=>{})
    refresh()
    const timer = setInterval(refresh,180000)
    return ()=>{controller.abort(); clearInterval(timer)}
  },[])
  react.useLayoutEffect(()=>{
    const saved = localStorage.getItem('darask-mode')
    if (saved && dsModes[saved]) {
      theme()?.setTheme(saved === 'paper' ? 'light' : saved)
      document.documentElement.dataset.studioMode = saved
    }
  },[mode])
  function change(next) {
    localStorage.setItem('darask-mode',next)
    document.documentElement.dataset.studioMode = next
    theme()?.setTheme(next === 'paper' ? 'light' : next)
    setMode(next)
  }
  return dsElement('div',{className:'ds-footer','data-wide':wide},
    wide && dsElement('div',{className:'ds-quotas','aria-label':'利用可能な残量とリセット日時（日本時間）'},
      dsElement('div',{className:'ds-quota ds-quota-head','aria-hidden':true},dsElement('span',null,'利用枠'),dsElement('span',null,'残り'),dsElement('span',null,'リセット（JST）')),
      Object.entries({...dsAgents,devin:'Devin'}).map(([id,label])=>{
        const entry = usage[id], windows = [...entry?.windows || []].sort((a,b)=>a.remaining-b.remaining)
        const main = windows[0]
        const when = row => row.reset ? (row.approximate?'約 ':'')+dsDate.format(new Date(row.reset)) : '日時未定'
        const percent = value => new Intl.NumberFormat('ja-JP',{maximumFractionDigits:1}).format(value)+'%'
        const detail = windows.map(w=>w.label+' · 残り '+percent(w.remaining)+' · '+when(w)+' JST').join('\n')
        return dsElement('div',{className:'ds-quota',key:id,title:detail || '利用枠を取得できません',tabIndex:0},
          dsElement('span',null,label),dsElement('span',null,main ? (entry.unavailable?'≈ ':'')+percent(main.remaining) : (entry?'取得不可':'…')),
          dsElement('time',{dateTime:main?.reset || undefined},main ? when(main) : '—'),
          detail && dsElement('span',{className:'ds-quota-detail',role:'tooltip'},detail))
      })),
    dsElement('div',{className:'ds-actions'},
      dsElement(dsUI.Button,{type:'button',size:'sm',onClick:()=>openSettings(),className:'ds-settings','aria-label':'設定',title:'設定',icon:dsElement(dsUI.IconSettingsOutlineMedium,{size:16})},wide?'設定':null),
      ...Object.entries(dsModes).map(([id,[icon,label]])=>dsElement(dsUI.Button,{key:id,type:'button',size:'sm',title:label,'aria-label':label,'aria-pressed':mode===id,onClick:()=>change(id),icon:dsElement(icon,{size:15})}))))
}
function DaraskOptions() {
  const [data,setData] = react.useState(null), [agent,selectAgent] = react.useState('codex')
  const [values,setValues] = react.useState({}), [notice,setNotice] = react.useState(''), [busy,setBusy] = react.useState(false)
  react.useEffect(()=>{
    const ctrl = new AbortController()
    smFetchOptions(false,ctrl.signal).then(d=>{setData(d);setValues(d.values)}).catch(e=>{if (!ctrl.signal.aborted) setNotice(e.message)})
    return ()=>ctrl.abort()
  },[])
  async function refresh() {
    setBusy(true);setNotice('モデル一覧を更新中…')
    try {setData(await smFetchOptions(true));setNotice('一覧を更新しました')}
    catch(e){setNotice(e.message)} finally{setBusy(false)}
  }
  async function save() {
    setBusy(true)
    try { await dsFetch('options',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({agent,values:values[agent] || {}})});setNotice('既定値を保存しました') }
    catch(e){setNotice(e.message)} finally{setBusy(false)}
  }
  return dsElement('section',{className:'ds-options'},dsElement('h2',null,'LLMサービス'),
    dsElement('small',null,'ここでは選択時の既定値を設定します。チャットごとの指定は入力欄のモデル選択から変更できます。'),
    dsElement('label',null,'使用するLLMサービス',dsElement('select',{'aria-label':'使用するLLMサービス',value:agent,disabled:busy,onChange:e=>{selectAgent(e.target.value);setNotice('')}},Object.entries(data?.services || smServices).map(([id,n])=>dsElement('option',{key:id,value:id},n)))),
    data && dsElement(ServiceOptions,{service:agent,cap:data.catalog[agent],value:values[agent] || {},disabled:busy,onChange:row=>{setValues({...values,[agent]:row});setNotice('')}}),
    dsElement('div',{className:'ds-model-actions'},dsElement(dsUI.Button,{type:'button',variant:'outline',onClick:refresh,disabled:busy},'一覧を更新'),
      dsElement(dsUI.Button,{type:'button',variant:'primary',onClick:save,disabled:busy || !data},'保存')),
    dsElement('div',{role:'status'},notice || (!data?'読み込み中…':'')))
}
