import {mkdir,readFile} from 'node:fs/promises'
import {join} from 'node:path'
import {fileURLToPath} from 'node:url'
import {resolveDshHome} from '@deepseek-ai/dsh-home-paths'
import {withFileLock, writeFileAtomic} from '@deepseek-ai/dsh-atomic-write'
import {entryListSchema} from '@deepseek-ai/cordis-plugin-include'
import yaml from 'js-yaml'

const home = resolveDshHome()
const config = join(home,'cordis.patch.yml')
const entry = {id:'darask-workspace-folders',name:fileURLToPath(new URL('./index.mjs',import.meta.url))}
await mkdir(home,{recursive:true})
await withFileLock(config, async () => {
  const original = await readFile(config,'utf8').catch(error => {
    if (error.code === 'ENOENT') return ''
    throw error
  })
  const options = {schema:entryListSchema}
  const rows = yaml.load(original,options) ?? []
  const found = rows.flatMap(row => row.insert ?? []).find(row => row.id === entry.id)
  if (found?.name === entry.name) return
  if (found) throw Error('別の作業フォルダプラグインが登録されています。')
  await writeFileAtomic(config,original.trimEnd()+'\n'+yaml.dump([{insert:[entry]}],options),{mode:0o600})
})
console.log('チャットの複数作業フォルダを有効にしました。')
