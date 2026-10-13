import { spawn } from 'node:child_process'

const lifecycle = process.env.npm_lifecycle_event ?? ''
const command = lifecycle.startsWith('references:') ? lifecycle.slice('references:'.length) : null
if (!command || !['setup', 'qualify', 'import-assistant', 'expand', 'status', 'resume', 'finalize', 'rollback'].includes(command)) {
  process.stderr.write('Run this helper through one of the npm references:* commands.\n')
  process.exit(2)
}

const child = spawn('uv', [
  'run', '--project', 'services/api', '--locked', 'drawable', 'references', command,
  ...process.argv.slice(2),
], { cwd: process.cwd(), stdio: 'inherit', env: process.env })

child.on('error', (error) => {
  process.stderr.write(`Could not start the reference workflow: ${error.message}\n`)
  process.exitCode = 2
})
child.on('exit', (code, signal) => {
  process.exitCode = code ?? (signal ? 2 : 0)
})
