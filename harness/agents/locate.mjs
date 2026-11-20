import { executable } from './cli.mjs'
console.log(await executable(process.argv[2]))
