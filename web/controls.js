// This module is embedded in the pinned settings client bundle by scripts/patch-ui.mjs.
const dsElement = react.createElement
const dsAgents = {muse:'Muse',grok:'Grok',cursor:'Cursor',codex:'Codex',claude:'Claude'}
const dsCliNames = {'devin-free':'Devin（無料のみ）',devin:'Devin（全モデル）',...dsAgents}
const dsModes = {light:['☀','ライトモード'],dark:['☾','ダークモード'],paper:['▤','ペーパーモード']}
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
      dsElement('button',{type:'button',onClick:()=>openSettings(),className:'ds-settings','aria-label':'設定',title:'設定'},wide?'⚙  設定':'⚙'),
      ...Object.entries(dsModes).map(([id,[icon,label]])=>dsElement('button',{key:id,type:'button',title:label,'aria-label':label,'aria-pressed':mode===id,onClick:()=>change(id)},icon))))
}
function DaraskOptions() {
  const [data,setData] = react.useState(null), [agent,selectAgent] = react.useState('codex')
  const [values,setValues] = react.useState({}), [notice,setNotice] = react.useState(''), [busy,setBusy] = react.useState(false)
  react.useEffect(()=>{
    const ctrl = new AbortController()
    dsFetch('options',{signal:ctrl.signal}).then(d=>{setData(d);setValues(d.values)}).catch(e=>{if (!ctrl.signal.aborted) setNotice(e.message)})
    return ()=>ctrl.abort()
  },[])
  const set = (key,value)=>{
    const row = {...values[agent]}
    if (value === '' || value === false) delete row[key]
    else row[key] = value
    if (key === 'model' && data.catalog[agent].byModel?.[value]) {
      const cap = data.catalog[agent].byModel[value]
      if (!cap.efforts.includes(row.effort)) delete row.effort
      if (!cap.fast) delete row.fast
    }
    setValues({...values,[agent]:row});setNotice('')
  }
  const save = async()=>{
    setBusy(true)
    try { await dsFetch('options',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({agent,values:values[agent]})});setNotice('保存しました') }
    catch(e){setNotice(e.message)} finally{setBusy(false)}
  }
  const isDevin = agent === 'devin' || agent === 'devin-free'
  const row = values[agent] || {}, cap = data?.catalog[agent], specific = cap?.byModel?.[row.model] || cap
  return dsElement('section',{className:'ds-options'},dsElement('h2',null,'ローカル CLI'),
    dsElement('label',null,'エージェント',dsElement('select',{value:agent,disabled:busy,onChange:e=>{selectAgent(e.target.value);setNotice('')}},Object.entries(dsCliNames).map(([id,n])=>dsElement('option',{key:id,value:id},n)))),
    cap && dsElement(react.Fragment,null,
      isDevin ? dsElement('label',null,'モデル',dsElement('select',{'aria-label':'モデル',value:row.model || '',onChange:e=>set('model',e.target.value)},
        dsElement('option',{value:''},'無料モデルを自動選択'),
        ...[true,false].map(free=>{const rows=(cap.variants || []).filter(v=>v.free===free);return rows.length ? dsElement('optgroup',{key:String(free),label:free?'無料':'その他の料金'},
          rows.map(v=>dsElement('option',{key:v.id,value:v.id},(free?'無料 · ':'')+v.label))):null})),
        dsElement('small',null,cap.unavailable?'Devin CLI でログインしてモデル一覧を取得してください。':(cap.variants?.find(v=>v.id===row.model)?.price || 'Devin が無料と表示しているモデルを使います。'))) :
      dsElement('label',null,'モデル',dsElement('input',{list:'ds-cli-models',value:row.model || '',placeholder:'CLI の既定値',onChange:e=>set('model',e.target.value)}),
        dsElement('datalist',{id:'ds-cli-models'},cap.models.map(m=>dsElement('option',{key:m,value:m})))),
      agent==='cursor' && dsElement('small',null,'エフォートと高速設定はモデル名で選択します。'),
      agent==='grok' && dsElement('small',null,'高速設定は Fast モデルで選択します。'),
      specific.efforts.length>0 && dsElement('label',null,'エフォート',dsElement('select',{value:row.effort || '',onChange:e=>set('effort',e.target.value)},['',...specific.efforts].map(v=>dsElement('option',{key:v,value:v},v || 'CLI の既定値')))),
      ['fast','ultracode'].filter(k=>specific[k] || (k==='ultracode' && cap[k])).map(k=>dsElement('label',{className:'ds-check',key:k},
        dsElement('input',{type:'checkbox',checked:!!row[k],onChange:e=>set(k,e.target.checked)}),k==='fast'?'高速モード':'Ultracode')),
      dsElement('button',{type:'button',className:'ds-save',onClick:save,disabled:busy},'保存')),
    dsElement('div',{role:'status'},notice || (!data?'読み込み中…':'')))
}
