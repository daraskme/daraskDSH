import assert from 'node:assert/strict'
import test from 'node:test'
import {mkdtemp,mkdir,symlink,rm} from 'node:fs/promises'
import {join} from 'node:path'
import {tmpdir} from 'node:os'
import {fileURLToPath} from 'node:url'
import {CliAdapter,parseReply,needsToolReplyRepair} from '../agents/index.mjs'

const tools = [{name:'bash',description:'Run a fixture command',inputSchema:{type:'object'}}]
const call = JSON.stringify({dsh_tool:'bash',arguments:{command:'printf "a {brace}"'}})

test('mixed prose requests require correction; examples remain text',()=>{
  const mixed='まず現状の正本を読んで、仕分けの土台にします。'+call
  assert.deepEqual(parseReply(mixed,tools),[{type:'text',text:mixed}])
  assert.equal(needsToolReplyRepair(mixed,tools),true)
  assert.equal(needsToolReplyRepair('調べます。{"dsh_tool":"bash","arguments":',tools),true)
  for (const sample of ['例です。\n```json\n'+call+'\n```','例です。\n~~~json\n'+call+'\n~~~','例は `'+call+'` です。','引用：\n> '+call]) {
    assert.equal(needsToolReplyRepair(sample,tools),false)
    assert.deepEqual(parseReply(sample,tools),[{type:'text',text:sample}])
  }
  for(const reply of [call,'['+call+']','```json\n'+call+'\n```',call+'\nWaiting for the tool result.']) {
    assert.equal(parseReply(reply,tools)[0].type,'tool-call')
  }
  assert.equal(parseReply(call+'\n'+call,tools).length,2)
  assert.throws(()=>parseReply('{"dsh_tool":"unknown","arguments":{}}',tools),/Unavailable/)
  assert.throws(()=>parseReply('{"dsh_tool":"bash","arguments":null}',tools),/Invalid/)
})

test('bridge retries malformed Devin output once and fails explicitly if still malformed',async t=>{
  const root=await mkdtemp(join(tmpdir(),'dsh-tool-replies-'))
  t.after(()=>rm(root,{recursive:true,force:true}))
  const bin=join(root,'bin');await mkdir(bin)
  await symlink(fileURLToPath(new URL('agents.py',import.meta.url)),join(bin,'devin'))
  const previous={PATH:process.env.PATH,DSH_HOME:process.env.DSH_HOME}
  process.env.PATH=bin+':'+process.env.PATH;process.env.DSH_HOME=join(root,'state')
  t.after(()=>{for(const [key,value] of Object.entries(previous)) {if(value===undefined)delete process.env[key];else process.env[key]=value}})
  async function collect(text) {
    const chunks=[]
    for await(const chunk of new CliAdapter().stream({model:'devin-free',tools,system:'Test',messages:[{role:'user',content:[{type:'text',text}]}]})) chunks.push(chunk)
    return chunks
  }
  const chunks=await collect('TEST_MIXED_TOOL')
  assert.equal(chunks.filter(c=>c.type==='tool-call-delta').length,1)
  assert.equal(chunks.some(c=>c.type==='text-delta'),false)
  assert.deepEqual(chunks.at(-1),{type:'finish',reason:{kind:'tool-calls'}})
  await assert.rejects(collect('TEST_MIXED_STUCK'),/このステップのツールは未実行/)
})
