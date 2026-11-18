function DaraskInstructions() {
  const [scope,setScope]=react.useState('global'), [key,setKey]=react.useState('AGENTS.md')
  const [catalog,setCatalog]=react.useState(null), [doc,setDoc]=react.useState(null), [body,setBody]=react.useState('')
  const [notice,setNotice]=react.useState(''), [saving,setSaving]=react.useState(false)
  const [newName,setNewName]=react.useState(''), [refresh,setRefresh]=react.useState(0)
  const [reloadDoc,setReloadDoc]=react.useState(0)
  const dirty=doc && body!==doc.body
  const editable=!saving && !!doc
  function canSwitch() {return !dirty || window.confirm('未保存の編集を破棄しますか？')}
  react.useEffect(()=>{
    const ctrl=new AbortController()
    dsFetch('instructions?'+new URLSearchParams({scope}),{signal:ctrl.signal}).then(setCatalog)
      .catch(e=>{if(!ctrl.signal.aborted)setNotice(e.message)})
    return ()=>ctrl.abort()
  },[scope,refresh])
  react.useEffect(()=>{
    const ctrl=new AbortController()
    dsFetch('instructions?'+new URLSearchParams({scope,key}),{signal:ctrl.signal}).then(d=>{
      if(ctrl.signal.aborted)return
      setDoc(d)
      const match=key.match(/^skills\/([^/]+)\/SKILL\.md$/)
      setBody(d.revision===null && match?'---\nname: '+match[1]+'\ndescription: \n---\n\n':d.body)
    }).catch(e=>{if(!ctrl.signal.aborted)setNotice(e.message)})
    return ()=>ctrl.abort()
  },[scope,key,reloadDoc])
  function select(nextScope,nextKey) {
    if(nextScope===scope && nextKey===key || !canSwitch())return
    if(nextScope!==scope)setCatalog(null)
    setDoc(null);setBody('');setNotice('');setScope(nextScope);setKey(nextKey)
  }
  async function save() {
    setSaving(true)
    try {
      const d=await dsFetch('instructions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({scope,key,body,revision:doc.revision})})
      setDoc(d);setBody(d.body);setNotice('保存しました');setRefresh(v=>v+1)
    } catch(e) {setNotice(e.message)} finally {setSaving(false)}
  }
  const e=dsElement
  return e('section',{className:'ds-options ds-instructions'},
    e('h2',null,'スキル・指示ファイル'),
    e('label',null,'適用範囲',e('select',{value:scope,disabled:saving,onChange:event=>select(event.target.value,'AGENTS.md')},
      (catalog?.scopes || []).map(s=>e('option',{key:s.id,value:s.id},s.title+(s.id==='global'?'':' · '+s.path))))),
    e('h3',null,'指示ファイル'),
    e('div',{className:'ds-doc-links'},(catalog?.documents || []).map(k=>e(dsUI.Button,{type:'button',size:'sm',key:k,'aria-pressed':key===k,disabled:saving,onClick:()=>select(scope,k)},k))),
    e('label',null,key,e('textarea',{value:body,disabled:!editable,spellCheck:false,rows:9,onChange:event=>{setBody(event.target.value);setNotice('')}})),
    doc && e('small',null,doc.path),
    e('div',{className:'ds-doc-links'},e(dsUI.Button,{variant:'primary',disabled:!editable || !dirty,onClick:save},'保存'),
      e(dsUI.Button,{variant:'outline',disabled:saving,onClick:()=>{if(canSwitch()){setDoc(null);setNotice('');setRefresh(v=>v+1);setReloadDoc(v=>v+1)}}},'再読み込み')),
    e('div',{role:'status'},notice),
    e('h3',null,'スキル一覧'),
    catalog && !catalog.complete && e('small',null,'一部のスキルを読み込めませんでした'),
    catalog?.skills.length===0 && e('small',null,'この範囲で使えるスキルはありません'),
    e('ul',null,(catalog?.skills || []).map(s=>e('li',{key:s.name},
      s.key?e(dsUI.Button,{size:'sm',disabled:saving,onClick:()=>select(scope,s.key)},s.name):e('strong',null,s.name),
      e('small',null,s.source),e('div',null,s.description)))),
    e('form',{onSubmit:event=>{event.preventDefault();const name=newName.trim();if(!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name)){setNotice('英小文字・数字・ハイフンで名前を入力してください');return}select(scope,'skills/'+name+'/SKILL.md')}},
      e('label',null,'新しいスキル名',e('input',{value:newName,placeholder:'my-skill',disabled:saving,onChange:event=>setNewName(event.target.value)})),
      e(dsUI.Button,{type:'submit',variant:'outline',disabled:saving || !newName.trim()},'スキルを開く・作成')))
}
