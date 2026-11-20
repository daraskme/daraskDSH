// Embedded in the Harness browse picker. Remount for each user-initiated pick.
function StudioWorkspacePicker(props) {
  const {Modal, Button} = _deepseek_ai_dsh_client_ui_primitives
  const h = react.createElement
  const [desktop, setDesktop] = react.useState(() =>
    /Linux/.test(navigator.userAgent) && !/Android|Mobile/.test(navigator.userAgent))
  const [notice, setNotice] = react.useState('')
  const current = react.useRef(props)
  const request = react.useRef(null)
  current.current = props

  const browse = () => {
    request.current?.abort()
    setDesktop(false)
  }
  const cancel = () => {
    request.current?.abort()
    current.current.onCancel()
  }
  react.useEffect(() => {
    if (!desktop) return
    const controller = new AbortController()
    request.current = controller
    async function pick() {
      try {
        const response = await fetch('/api/darask.pickWorkspace', {
          method:'POST', headers:{'Content-Type':'application/json'}, body:'{}', signal:controller.signal
        })
        const result = await response.json()
        if (!response.ok) throw Error(result.error || 'Ubuntu の選択画面を開けません')
        if (controller.signal.aborted) return
        if (result.path === null) current.current.onCancel()
        else current.current.onPicked(result.path)
      } catch (error) {
        if (controller.signal.aborted) return
        setNotice(error.message)
        setDesktop(false)
      }
    }
    const start = setTimeout(pick, 0)
    return () => {clearTimeout(start); controller.abort()}
  }, [desktop])

  if (!desktop) return h(DirectoryBrowser, {
    open:true, busy:props.busy, listDirectory:props.listDirectory, createDirectory:props.createDirectory,
    t:props.t, onOpen:props.onPicked, onClose:props.onCancel, desktopNotice:notice,
    openDesktop:() => {setNotice(''); setDesktop(true)}
  })
  return h(Modal, {
    open:true, title:'ワークスペースを選択', closeLabel:props.t('browser.cancel'),
    onClose:props.busy ? () => {} : cancel,
    footer:h(react.Fragment,null,
      h(Button,{variant:'outline',onClick:cancel,disabled:props.busy},props.t('browser.cancel')),
      h(Button,{variant:'primary',onClick:browse,disabled:props.busy},'この画面で選択'))
  }, h('p',{role:'status'},'Ubuntu のデスクトップでフォルダーを選んでください。'))
}
