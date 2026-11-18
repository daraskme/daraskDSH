window.__ModuleLoader__.load({
  id:'darask-workspace-folders',
  factory(require) {
    const React = require('react')
    const {Button, Modal, IconFolderOpenRegular, IconFolderCloseRegular, IconPlusOutlineRegular, IconCloseOutlineRegular} = require('@deepseek-ai/dsh-client-ui-primitives')
    const h = React.createElement

    const styles = `
      .dsh-folders-modal.dsh-folders-modal {width:min(480px,calc(100vw - 24px));max-height:calc(100dvh - 24px)}
      .dsh-folders-body {display:grid;gap:18px;overflow-y:auto;max-height:55dvh;min-width:0}
      .dsh-folders-section {display:grid;gap:8px;min-width:0}
      .dsh-folders-label {font-size:12px;font-weight:500;color:var(--dsw-alias-label-secondary)}
      .dsh-folders-list {display:grid;gap:2px;padding:4px;background:var(--dsw-alias-bg-module-platform);border-radius:var(--dsw-radius-md)}
      .dsh-folders-row {display:flex;align-items:center;gap:9px;padding:6px 8px;min-width:0}
      .dsh-folders-row svg {flex-shrink:0;color:var(--dsw-alias-label-secondary)}
      .dsh-folders-path {min-width:0;flex:1;display:grid;gap:2px;overflow-wrap:anywhere}
      .dsh-folders-path small {color:var(--dsw-alias-label-secondary);font-size:12px}
      .dsh-folders-form {display:flex;align-items:center;gap:8px;min-width:0}
      .dsh-folders-input {box-sizing:border-box;width:100%;min-width:0;border:1px solid var(--dsw-alias-border-l3);border-radius:var(--dsw-radius-md);padding:7px 9px;font:inherit;color:var(--dsw-alias-label-primary);background:var(--dsw-alias-bg-module-platform)}
      .dsh-folders-body [role=alert] {color:var(--dsw-alias-state-error-primary);margin:0}
      .dsh-folders-trigger {white-space:nowrap}
    `
    function FolderRow({folder,primary,remove,disabled}) {
      const name = folder.split(/[\\/]/).filter(Boolean).at(-1) || folder
      return h('div',{className:'dsh-folders-row'},
        h(primary ? IconFolderOpenRegular : IconFolderCloseRegular,{size:18,'aria-hidden':true}),
        h('span',{className:'dsh-folders-path'},h('span',null,name),h('small',null,folder)),
        remove && h(Button,{size:'sm','aria-label':'外す',title:name+' を外す',disabled,onClick:remove,
          icon:h(IconCloseOutlineRegular,{size:14})}))
    }
    function FolderDialog({sessionId, close, renderPicker}) {
      const [state,setState] = React.useState(null)
      const [folders,setFolders] = React.useState([])
      const [path,setPath] = React.useState('')
      const [notice,setNotice] = React.useState('')
      const [saving,setSaving] = React.useState(false)
      const [picking,setPicking] = React.useState(false)
      const url = '/api/darask.workspaceFolders?session='+encodeURIComponent(sessionId)
      React.useEffect(() => {
        const ctrl = new AbortController()
        fetch(url,{signal:ctrl.signal}).then(async res => {
          const data = await res.json()
          if (!res.ok) throw Error(data.error)
          if (!ctrl.signal.aborted) {setState(data); setFolders(data.folders)}
        }).catch(error => {if (!ctrl.signal.aborted) setNotice(error.message)})
        return () => ctrl.abort()
      },[url])
      function add(value) {
        if (value && value !== state.primary) setFolders(previous => [...new Set([...previous,value])])
      }
      async function save() {
        setSaving(true)
        setNotice('')
        try {
          const selected = path.trim() ? [...folders,path.trim()] : folders
          const res = await fetch(url,{method:'POST', headers:{'content-type':'application/json'},
            body:JSON.stringify({revision:state.revision,folders:selected})})
          const data = await res.json()
          if (!res.ok) throw Error(data.error)
          close()
        } catch (error) { setNotice(error.message); setSaving(false) }
      }
      const candidates = state?.workspaces.filter(ws => ws.path !== state.primary && !folders.includes(ws.path)) ?? []
      if (picking) return renderPicker({
        open:true,busy:false,onPicked:value=>{add(value);setPicking(false)},onCancel:()=>setPicking(false),
        onError:message=>{setNotice(message);setPicking(false)}
      })
      return h(Modal,{open:true,className:'dsh-folders-modal',title:'このチャットの作業フォルダ',closeLabel:'閉じる',onClose:saving ? ()=>{} : close,
        footer:h(React.Fragment,null,h(Button,{variant:'outline',disabled:saving,onClick:close},'キャンセル'),
          h(Button,{variant:'primary',disabled:!state || saving,onClick:save},saving ? '保存中…' : '保存'))},
        h('div',{className:'dsh-folders-body','aria-busy':!state && !notice},
          !state && !notice && h('span',{role:'status'},'読み込み中…'),
          state && h(React.Fragment,null,
            h('section',{className:'dsh-folders-section','aria-label':'メインフォルダ'},
              h('span',{className:'dsh-folders-label'},'メインフォルダ'),
              h('div',{className:'dsh-folders-list'},h(FolderRow,{folder:state.primary,primary:true}))),
            h('section',{className:'dsh-folders-section','aria-label':'追加フォルダ'},
              h('span',{className:'dsh-folders-label'},'追加フォルダ'),
              renderPicker && h('div',null,h(Button,{type:'button',size:'sm',variant:'outline',disabled:saving,
                icon:h(IconFolderOpenRegular,{size:15}),onClick:()=>setPicking(true)},'フォルダを選択…')),
              folders.length>0 && h('div',{className:'dsh-folders-list'},folders.map(folder=>
                h(FolderRow,{key:folder,folder,disabled:saving,remove:()=>setFolders(previous=>previous.filter(x=>x!==folder))}))),
              candidates.length>0 && h('select',{className:'dsh-folders-input','aria-label':'登録済みワークスペースから追加',value:'',disabled:saving,onChange:e=>add(e.target.value)},
                h('option',{value:''},'登録済みワークスペースから追加…'),
                candidates.map(ws => h('option',{value:ws.path,key:ws.path},ws.title+' · '+ws.path))),
              h('form',{className:'dsh-folders-form',onSubmit:event=>{event.preventDefault();add(path.trim());setPath('')}},
                h('input',{'aria-label':'フォルダの絶対パス',placeholder:'フォルダの絶対パス',className:'dsh-folders-input',
                  value:path,disabled:saving,onChange:e=>setPath(e.target.value)}),
                h(Button,{type:'submit',variant:'outline',disabled:saving || !path.trim(),icon:h(IconPlusOutlineRegular,{size:14})},'追加')))),
          notice && h('p',{role:'alert'},notice)))
    }
    function FolderButton({sessionId,renderSlot,pickerName,hasPicker}) {
      const [open,setOpen] = React.useState(false)
      return h(React.Fragment,null,
        h(Button,{type:'button',size:'sm',className:'dsh-folders-trigger',title:'このチャットの作業フォルダ',
          'aria-haspopup':'dialog','aria-expanded':open,icon:h(IconFolderOpenRegular,{size:16}),onClick:()=>setOpen(true)},'作業フォルダ'),
        open && h(FolderDialog,{key:sessionId,sessionId,close:()=>setOpen(false),
          renderPicker:hasPicker() ? owner=>renderSlot(pickerName,owner) : null}))
    }
    return {inject:['slots'],apply(ctx) {
      const style = document.createElement('style')
      style.textContent = styles
      document.head.append(style)
      ctx.effect(()=>()=>style.remove(),'Workspace folder styles')
      for (const [name,pickerName] of [
        ['conversation.hero.options','darask.folders.hero.picker'],
        ['conversation.session.header.actions','darask.folders.header.picker']
      ]) {
        ctx.slots.inject(name,()=>ctx.slots.register({
          name,id:'workspace-folders',children:{[pickerName]:{kind:'single',scope:'root'}},
          inject:sessionId=>({sessionId,pickerName,hasPicker:()=>ctx.slots.entries(pickerName).length>0})
        }, FolderButton))
      }
    }}
  }
})
