import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import yaml from 'js-yaml'
import { entryListSchema } from '@deepseek-ai/cordis-plugin-include'
const base = new URL('../',import.meta.url)
const get = path=>readFile(new URL(path,base),'utf8')
const controls = (await Promise.all(['web/model-options.js','web/controls.js','web/instructions.js'].map(get))).join('\n'), css = await get('web/controls.css')
async function patch(name,transform,entry = 'dsh-client-ui-'+name+'/lib/client.js') {
  const file = fileURLToPath(new URL('node_modules/@deepseek-ai/'+entry,base))
  const original = file+'.darask-upstream'
  let source
  try { source = await readFile(original,'utf8') }
  catch(e) { if(e.code!=='ENOENT') throw e; source=await readFile(file,'utf8'); await writeFile(original,source,{flag:'wx'}) }
  function replace(needle,value) {
    const at = source.indexOf(needle)
    if (at<0 || source.indexOf(needle,at+needle.length)!==-1) throw Error(name+' unexpected pinned source: '+needle.slice(0,70))
    source = source.slice(0,at) + value + source.slice(at+needle.length)
  }
  function between(start,end,value) {
    const begin = source.indexOf(start), finish = source.indexOf(end,begin)
    if(begin<0 || finish<0) throw Error(name+' missing patch boundary')
    replace(source.slice(begin,finish),value)
  }
  transform(replace,between)
  if(await readFile(file,'utf8')!==source) await writeFile(file,source)
}
// Web conversations own their compaction service inside the selected preset;
// the host-level policy used by headless sessions does not reach those services.
const compaction = yaml.load(await get('config/agents.yml'),{schema:entryListSchema})
  .find(row=>row.id==='compaction-basic')?.config
if (!compaction) throw Error('Missing compaction-basic policy in config/agents.yml')
const presetCompaction = yaml.dump({config:compaction}).trimEnd().split('\n').map(line=>'                '+line).join('\n')+'\n'
for (const preset of ['standard','minimal','ptc','cordis']) {
  await patch('compaction-preset-'+preset,replace=>{
    const entry = "                name: '@deepseek-ai/dsh-compaction-basic'\n"
    replace(entry,entry+presetCompaction)
  },'dsh-web-app/presets/'+preset+'.patch.yml')
}
// The GUI's catalog only lists service entry points. Explicit per-chat CLI
// selections are validated by the adapter, as in the LLM registry's routing API.
await patch('session-controller',(replace)=>{
  const start = 'async function modelAvailable(ctx, selection) {'
  replace(start,start+`
    if (selection.provider === "local-cli" && selection.model.includes("@")) {
      try { await ctx.llm.resolveModelInfo(selection.provider,selection.model); return true }
      catch { return false }
    }`)
},'dsh-api-session-controller/lib/index.js')
await patch('settings-general',(replace,between)=>{
  replace('function SettingsRoot(props) {',controls+'\nfunction SettingsRoot(props) {')
  replace('onboardingActive ? onboardingSteps.find((step) => !completedOnboarding.has(step.id)) : void 0','void 0')
  between('renderSlot("settings.launcher", {','(0, react_jsx_runtime.jsx)(_deepseek_ai_dsh_client_ui_primitives.ConnectionIndicator,',
    'dsElement(DaraskFooter,{wide,openSettings:actions.open,theme:props.studioTheme}),\n')
  replace('const shellInjected = () => ({','const shellInjected = () => ({studioTheme:()=>ctx.get("theme"),')
  const nav = 'function navIcon(id) {'
  replace(nav,nav+`
    if (id === "darask-cli") return dsElement(dsUI.IconSlidersTwoOutlineRegular,{size:16});
    if (id === "darask-instructions") return dsElement(dsUI.IconSkillOutlineRegular,{size:16});
    if (id === "schedules") return dsElement(dsUI.IconClockOutlineRegular,{size:16});`)
  const section='ctx.slots.inject("settings.section", () => ctx.slots.register({'
  replace(section,`ctx.slots.inject("settings.section",()=>ctx.slots.register({name:"settings.section",id:"darask-cli",order:12,label:()=>"LLMサービス"},DaraskOptions));
    ctx.slots.inject("settings.section",()=>ctx.slots.register({name:"settings.section",id:"darask-instructions",order:13,label:()=>"スキル・指示ファイル"},DaraskInstructions));
    const style = document.createElement("style");
    style.textContent = ${JSON.stringify(css)};
    document.head.append(style);
    ctx.effect(()=>()=>style.remove(),"Studio styles");
    `+section)
})
await patch('sidebar',(replace,between)=>{
  replace('id === "plugins" || !unpinned.includes(id)','!["plugins","schedules"].includes(id) && !unpinned.includes(id)')
  between('}, id)), hasEntryActions && (0, react_jsx_runtime.jsx)(PanelMoreMenu, {',']\n\t\t\t\t\t}),', '}, id))')
})
await patch('layout',(replace)=>{
  replace('function AppFrame({',`const StudioMain = react.createContext(null);
    function StudioPanel({panel,close}) {
      const render = react.useContext(StudioMain);
      return render && react.createElement("div",{className:"ds-settings-panel"},render("main",{closeStudioSettings:close}, {entryKey:panel}));
    }
    function AppFrame({`)
  replace('children: sidebar\n','children: react.createElement(StudioMain.Provider,{value:renderSlot},sidebar)\n')
  replace('function apply(ctx) {',`function apply(ctx) {
    ctx.slots.inject("settings.section",()=>ctx.slots.register({name:"settings.section",id:"schedules",order:16,label:()=>"自動化タスク"},({close})=>react.createElement(StudioPanel,{panel:"schedules",close})));
    ctx.slots.inject("settings.plugins.tab",()=>ctx.slots.register({name:"settings.plugins.tab",id:"manage",order:-10,label:()=>"管理"},()=>react.createElement(StudioPanel,{panel:"plugins"})));
  `)
})

await patch('settings-plugins',(replace)=>{
  replace('label: () => t("nav"),','label: () => t("nav") === "組み込みプラグイン" ? "プラグイン" : t("nav"),')
})
await patch('conversation',(replace,between)=>{
  between('function HeroShell({ t, renderSlot, children }) {','//#endregion\n\t\t//#region lib/types/client/skeleton/ConversationContent.js',
    'function HeroShell({children}) { return react.createElement("div",{className:HeroShell_module_css_default.root},children) }\n\t\t')
  const anchor = 'const [pickerOpen, setPickerOpen] = (0, react.useState)(false);'
  replace(anchor,anchor+`
    const [studioChoice, setStudioChoice] = react.useState(null);
    const [modelBusy, setModelBusy] = react.useState(false);
    const [openingWorkspace, setOpeningWorkspace] = react.useState(false);
    const chosen = studioChoice?.sessionId === sessionId ? studioChoice?.selection : undefined;`)
  const variant = 'variant: hero ? "hero" : "composer",'
  replace(variant,variant+`
    studioModel:hero ? {draft:chosen,locked:openingWorkspace,onBusy:setModelBusy,
      onSelect:selection => setStudioChoice({sessionId,selection})} : undefined,`)
  replace('function WorkspaceChip({ buttonRef, label,','function WorkspaceChip({ buttonRef, disabled, label,')
  replace('ref: buttonRef,\n\t\t\t\ttype: "button",','ref: buttonRef, disabled,\n\t\t\t\ttype: "button",')
  replace('buttonRef: pickerAnchor,','buttonRef: pickerAnchor, disabled:modelBusy || openingWorkspace,')
  replace('selectWorkspace(workspaceId).catch(() => {','setOpeningWorkspace(true);\nselectWorkspace(workspaceId,chosen).finally(() => setOpeningWorkspace(false)).catch(() => {')
  replace('selectWorkspace: (workspaceId) => workspaceNavigation.openWorkspace','selectWorkspace: (workspaceId,studioModel) => workspaceNavigation.openWorkspace')
  const end = '})\n\t\t\t\t})\n\t\t\t}, ConversationContent);'
  replace(end,'}, {studioModel})\n\t\t\t\t})\n\t\t\t}, ConversationContent);')
  const blocked = '} : !inert && composerBlock !== void 0 ? {'
  replace(blocked,'} : hero && (modelBusy || openingWorkspace) ? {blocked:{reason:"設定を反映中…"},placeholder:"設定を反映中…"} : !inert && composerBlock !== void 0 ? {')
  replace('variant, disabled: inert = false,','variant, studioModel, disabled: inert = false,')
  replace('const modelSeatLocked = removed || inert || !live;',
    'const modelSeatLocked = removed || (studioModel ? studioModel.locked : inert || !live);')
  replace('sessionId === void 0 ? null : renderSlot("conversation.input.model", { locked: modelSeatLocked })',
    'renderSlot("conversation.input.model", {locked:modelSeatLocked,studioModel})')
  replace('"conversation.input.model": {\n\t\t\t\t\t\tkind: "single",\n\t\t\t\t\t\tscope: "session"',
    '"conversation.input.model": {kind:"single",scope:"session-maybe"')
})

const draftModel = (await Promise.all(['web/model-options.js','web/session-model.js'].map(get))).join('\n')
await patch('model-selection',(replace,between)=>{
  between('function ModelSelect({ locked, available, directory, load, select, resolveText, t, renderSlot }) {',
    '//#endregion\n\t\t//#region lib/types/client/locales.js',draftModel+'\n\t\t')
  const injected = 'const directory = models.directoryFor(sessionId);'
  replace(injected,'if (sessionId === undefined) return studioDraftActions(models,ctx);\n'+injected)
})
await patch('schedule',(replace)=>{
  const from = 'const { useCatalog, onNewTask, onClearEnded, onRetry, onCreateTask, t } = props;'
  replace(from, 'const { useCatalog, onClearEnded, onRetry, onCreateTask, t } = props;\nconst onNewTask = () => { props.closeStudioSettings?.(); props.onNewTask() };')
})

await patch('workspace',(replace)=>{
  replace('"remote.directoryPicker",\n\t\t\t"layout",','"remote.directoryPicker",\n\t\t\t"remote.session",\n\t\t\t"layout",')
  const render = 'renderDirectoryFlow: (owner) => renderSlot("conversation.hero.workspace.directoryFlow", owner),'
  replace(render,'preferDirectory: true,\n'+render)
  replace('addOnly = false, onBusyChange','addOnly: forceAdd = false, preferDirectory = false, onBusyChange')
  const available = 'const flowAvailable = useDirectoryFlow((occupied) => occupied);'
  replace(available,available+'\nconst addOnly = forceAdd || (preferDirectory && flowAvailable);')
  const connect = 'sessionId = await this.connectBlank(workspaceId, reuse);'
  replace(connect,connect+'\nif (!navigation.aborted) await this.applyStudioModel(sessionId,reuse.studioModel);')
  const restore = 'if (sessionId !== void 0 && !navigation.aborted) this.replaceMain(sessionId, navigation, "preserve");'
  replace(restore,`if (sessionId !== void 0 && !navigation.aborted) {
    await this.applyStudioModel(sessionId);
    if (!navigation.aborted) this.replaceMain(sessionId,navigation,"preserve");
  }`)
  const open = 'openWorkspace(workspaceId, beforeOpen, options = {}) {'
  replace(open,`async applyStudioModel(sessionId, selected) {
    const models = this.ctx.get("modelDirectories");
    const draft = models?.studioDraft;
    const selection = selected ?? draft;
    if (!selection) return;
    const result = await this.ctx.remote.session.selectModel({sessionId,...selection});
    if (!result.ok) throw Error(result.error.message);
    if (models && draft === models.studioDraft) models.studioDraft = undefined;
  }
  `+open)
})
const picker = await get('web/workspace-picker.js')
await patch('directory-picker-browse',(replace,between)=>{
  const registration = 'ctx.slots.inject("conversation.hero.workspace.directoryFlow", () => ctx.slots.inject("sidebar.workspaces.directoryFlow", function* () {'
  replace(registration,`for (const name of ["darask.folders.hero.picker","darask.folders.header.picker"]) {
    ctx.slots.inject(name,()=>ctx.slots.register({name,inject:injected},BrowseDirectoryFlow));
  }
  `+registration)
  between('function BrowseDirectoryFlow(props) {','//#endregion\n\t\t//#region lib/types/client/index.js',
    picker+'\nfunction BrowseDirectoryFlow(props) { return props.open ? react.createElement(StudioWorkspacePicker,props) : null }\n\t\t')
  replace('function DirectoryBrowser({ open, listDirectory, createDirectory, onOpen, onClose, busy, t })',
    'function DirectoryBrowser({ open, listDirectory, createDirectory, onOpen, onClose, busy, t, openDesktop, desktopNotice })')
  const footer='className: DirectoryBrowser_module_css_default.footerBar,\n\t\t\t\t\t\t\tchildren: ['
  replace(footer,footer+'\nreact.createElement(_deepseek_ai_dsh_client_ui_primitives.Button,{variant:"outline",onClick:openDesktop,disabled:parentInert},"Ubuntu で開く"),')
  const crumb='(0, react_jsx_runtime.jsx)("div", {\n\t\t\t\t\t\t\t\tclassName: DirectoryBrowser_module_css_default.crumbBar,'
  replace(crumb,'desktopNotice && react.createElement("div",{role:"status"},desktopNotice), '+crumb)
})
