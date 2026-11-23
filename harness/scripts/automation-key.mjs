// Manages the /automation/v1 Bearer key for studio.darask.me.
// Only the key's sha256 is written to the studio env file — the raw key is printed exactly once.
// Usage:
//   node scripts/automation-key.mjs issue    # generate a new key (invalidates the previous one)
//   node scripts/automation-key.mjs status   # whether a key is configured
//   node scripts/automation-key.mjs revoke   # remove the key hash (disables the API key)
import {createHash, randomBytes} from 'node:crypto'
import {mkdir, readFile, writeFile} from 'node:fs/promises'
import {homedir} from 'node:os'
import {dirname, join} from 'node:path'

const ENV = process.env.STUDIO_ENV || join(homedir(), '.config/darask-dsh/studio.env')
const NAME = 'AUTOMATION_API_KEY_HASH'

const command = process.argv[2]
if (!['issue', 'status', 'revoke'].includes(command)) {
  console.error('用法: node scripts/automation-key.mjs <issue|status|revoke>')
  process.exit(2)
}

const lines = (await readFile(ENV, 'utf8').catch(() => '')).split('\n')
const rest = lines.filter(line => !line.startsWith(NAME + '='))
const configured = lines.some(line => new RegExp('^' + NAME + '=.').test(line))

if (command === 'status') {
  console.log(configured ? '已配置 API key（仅存哈希，原 key 无法找回）' : '未配置 API key：' + ENV)
  process.exit(0)
}

await mkdir(dirname(ENV), {recursive:true})

if (command === 'revoke') {
  await writeFile(ENV, rest.join('\n').replace(/\n{3,}/g, '\n\n'))
  console.log(configured ? '已撤销 API key（重启 darask-dsh.service 后生效）' : '本来就没有配置 API key')
  process.exit(0)
}

// issue
const key = 'dshak_' + randomBytes(24).toString('base64url')
const hash = createHash('sha256').update(key).digest('hex')
await writeFile(ENV, [...rest.filter(line => line.trim()), NAME + '=' + hash, ''].join('\n'))
console.log('新的 /automation/v1 API key（只显示这一次，请立刻保存）:\n\n  ' + key + '\n')
console.log('已把它的 sha256 写入 ' + ENV + '（' + NAME + '）；旧 key 已失效。')
console.log('重启后生效: systemctl --user restart darask-dsh.service')
