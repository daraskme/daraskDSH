window.__ModuleLoader__.load({
  id:'darask-dsh-comfyui',
  factory(require) {
    const React = require('react')
    function Editor() {
      return React.createElement('iframe',{
        title:'ComfyUI ワークフローエディター', src:'/comfy/',
        style:{width:'100%',height:'100%',minHeight:'320px',display:'block',border:0}
      })
    }
    function Icon() {
      return React.createElement('svg',{viewBox:'0 0 24 24',width:18,height:18,fill:'none',stroke:'currentColor',strokeWidth:1.6,'aria-hidden':true},
        React.createElement('path',{d:'M7 8v3h10V8M12 11v5'}),
        [[4,3],[14,3],[9,16]].map(([x,y])=>React.createElement('rect',{key:x,x,y,width:6,height:5,rx:1.3})))
    }
    return {inject:['slots'],apply(ctx) {
      ctx.slots.inject('main',()=>ctx.slots.register({name:'main',key:'comfyui'},Editor))
      ctx.slots.inject('sidebar.panellist',()=>ctx.slots.register({
        name:'sidebar.panellist',id:'comfyui',order:20,label:()=>'ComfyUI'
      },Icon))
    }}
  }
})
