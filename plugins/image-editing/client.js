window.__ModuleLoader__.load({
  id:'dsh-image-editing',
  factory(require) {
    const {h,Ph3Panel,ImageIcon,registerPanel} = require('dsh-media-shared')

    function EditPanel() {
      return h(Ph3Panel, {editing:true})
    }

    return {inject:['slots'],apply(ctx) {
      registerPanel(ctx, 'ph3-image-edit', 26, 'P-H3 画像編集', EditPanel, ImageIcon)
    }}
  }
})
