import { spawn } from 'node:child_process'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const args = process.argv.slice(2)
if (args.includes('--help') || args.includes('-h')) {
  process.stdout.write('Usage: npm run search:prepare -- [--data-root datasets] [--snapshot reports/private/curation-snapshots/<id>.json]\nDownloads pinned frozen models and indexes the latest saved educational prototype batch.\n')
  process.exit(0)
}
let dataRoot = path.join(root, 'datasets')
let snapshot
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--data-root' && args[i + 1]) dataRoot = path.resolve(root, args[++i])
  else if (args[i] === '--snapshot' && args[i + 1]) snapshot = args[++i]
  else throw new Error(`Unknown or incomplete argument: ${args[i]}`)
}

async function run(command, commandArgs) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, { cwd: root, stdio: 'inherit' })
    child.once('error', reject)
    child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`${command} failed (${signal ?? code}).`)))
    const stop = () => child.kill('SIGTERM')
    process.once('SIGINT', stop)
    process.once('SIGTERM', stop)
    child.once('close', () => { process.off('SIGINT', stop); process.off('SIGTERM', stop) })
  })
}

try {
  await run('uv', ['sync', '--project', 'ml', '--locked', '--extra', 'cpu'])
  await run(path.join(root, 'ml/.venv/bin/python'), ['-m', 'drawable_ml.prototype_worker', 'prepare', '--data-root', dataRoot, ...(snapshot ? ['--snapshot', snapshot] : [])])
  process.stdout.write('Your reviewed references are ready. Run npm run search:local to open the drawing app.\n')
} catch (error) {
  process.stderr.write(`Local search setup failed: ${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 2
}
