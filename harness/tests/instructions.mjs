import assert from 'node:assert/strict'
import {mkdtemp,mkdir,readFile,rm,symlink,writeFile} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import test from 'node:test'
import {instructionsApi} from '../web/instructions.mjs'

test('Studio edits native instructions and skills with revision and path checks',async t=>{
  const root=await mkdtemp(join(tmpdir(),'dsh-settings-'))
  t.after(()=>rm(root,{recursive:true,force:true}))
  const home=join(root,'profile'), repo=join(root,'repo'), cwd=join(repo,'project')
  await Promise.all([mkdir(home),mkdir(join(repo,'.git'),{recursive:true}),mkdir(cwd,{recursive:true})])
  const env={DSH_HOME:home,DSH_AGENTS_HOME:join(root,'shared')}
  for(const [k,v] of Object.entries(env)) {
    const old=process.env[k]
    process.env[k]=v
    t.after(()=>{if(old===undefined)delete process.env[k];else process.env[k]=old})
  }
  const api=instructionsApi({workspaceRegistry:{list:()=>[{id:'work',title:'test',path:cwd}]},
    skills:{list:async()=>[]},get:()=>undefined,logger:console,effect:f=>t.after(f())})
  const url='http://localhost/api/darask.instructions'
  const get=(q={})=>api(new Request(url+'?'+new URLSearchParams(q)))
  const save=(key,body,revision=null,scope='work')=>api(new Request(url,{method:'POST',
    headers:{'content-type':'application/json'},body:JSON.stringify({scope,key,body,revision})}))
  const original=await save('AGENTS.md','global',null,'global')
  assert.equal((await readFile(join(home,'AGENTS.md'),'utf8')),'global')
  await assert.rejects(save('AGENTS.md','stale',null,'global'),/他の編集/)
  await save('AGENTS.md','updated',original.revision,'global')
  await save('CLAUDE.local.md','project')
  assert.equal(await readFile(join(cwd,'CLAUDE.local.md'),'utf8'),'project')
  const body='---\nname: check-skill\ndescription: Test the catalog\n---\n\nRun the test.\n'
  const key='skills/check-skill/SKILL.md'
  await save(key,body)
  assert.equal(await readFile(join(repo,'.dsh',key),'utf8'),body)
  assert.deepEqual((await get({scope:'work'})).skills.map(s=>[s.name,s.key]),[['check-skill',key]])
  await assert.rejects(save('skills/../SKILL.md',body),/無効/)
  await assert.rejects(save('skills/bad/SKILL.md','no frontmatter'),/YAML/)
  await assert.rejects(get({scope:'/tmp'}),/未登録/)
  await writeFile(join(root,'elsewhere'),'private')
  await symlink(join(root,'elsewhere'),join(cwd,'CLAUDE.md'))
  await assert.rejects(get({scope:'work',key:'CLAUDE.md'}))
  await symlink(root,join(repo,'.dsh/skills/other'))
  await assert.rejects(save('skills/other/SKILL.md',body),/リンク/)
})
