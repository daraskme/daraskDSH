window.__ModuleLoader__.load({
  id:'darask-dsh-comfyui',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const {IconRightUpOutlineRegular} = require('@deepseek-ai/dsh-client-ui-primitives')
    const editorUrl = '/comfy/'
    function Editor() {
      return h('section',{className:'dsh-comfy-panel'},
        h('header',null,h('h1',null,h(Icon),'ComfyUI'),
          h('a',{href:editorUrl,target:'_blank',rel:'noopener'},'別タブで開く',h(IconRightUpOutlineRegular,{size:14}))),
        h('iframe',{title:'ComfyUI ワークフローエディター',src:editorUrl}))
    }
    function Icon() {
      return React.createElement('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        React.createElement('path',{d:'M7 8v3h10V8M12 11v5'}),
        [[4,3],[14,3],[9,16]].map(([x,y])=>React.createElement('rect',{key:x,x,y,width:6,height:5,rx:1.3})))
    }
    return {inject:['slots'],apply(ctx) {
      const style = document.createElement('style')
      style.textContent = `
        .dsh-comfy-panel{display:flex;flex-direction:column;height:100%;width:100%;min-width:0;min-height:0;background:var(--dsw-alias-bg-base);color:var(--dsw-alias-label-primary)}
        .dsh-comfy-panel header{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:28px;padding:10px 16px;border-bottom:1px solid var(--dsw-alias-border-l3)}
        .dsh-comfy-panel h1{display:flex;align-items:center;gap:8px;font:inherit;font-weight:550;margin:0}
        .dsh-comfy-panel a{display:inline-flex;align-items:center;gap:4px;color:inherit;font-size:13px}
        .dsh-comfy-panel iframe{display:block;flex:1;min-height:0;width:100%;border:0}
        @media(max-width:650px){.dsh-comfy-panel header{padding:8px 12px}}
      `
      document.head.append(style)
      ctx.effect(()=>()=>style.remove(),'ComfyUI panel styles')
      ctx.slots.inject('main',()=>ctx.slots.register({name:'main',key:'comfyui'},Editor))
      ctx.slots.inject('sidebar.panellist',()=>ctx.slots.register({
        name:'sidebar.panellist',id:'comfyui',order:20,label:()=>'ComfyUI'
      },Icon))
    }}
  }
})
