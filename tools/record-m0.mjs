import { spawn } from 'node:child_process'
import { mkdir, lstat, readFile, realpath } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { chromium } from 'playwright'

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, '..')
const WEB_ORIGIN = 'http://127.0.0.1:5173'
const API_ORIGIN = 'http://127.0.0.1:8000'

function parseArgs(argv) {
  let dataRoot = path.join(REPOSITORY_ROOT, 'datasets')
  let headless = false
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--data-root') {
      dataRoot = path.resolve(REPOSITORY_ROOT, argv[index + 1])
      index += 1
    } else if (argv[index] === '--headless') {
      headless = true
    } else if (argv[index] === '--help' || argv[index] === '-h') {
      process.stdout.write('Usage: npm run m0:record -- [--data-root datasets] [--headless]\n')
      process.exit(0)
    } else {
      throw new Error(`Unknown option: ${argv[index]}`)
    }
  }
  return { dataRoot, headless }
}

function start(command, args, env) {
  const child = spawn(command, args, {
    cwd: REPOSITORY_ROOT,
    env: { ...process.env, ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let output = ''
  const collect = (chunk) => { output = `${output}${chunk.toString()}`.slice(-5000) }
  child.stdout.on('data', collect)
  child.stderr.on('data', collect)
  return { child, get output() { return output } }
}

async function waitFor(url, server, label) {
  const deadline = Date.now() + 45_000
  let lastStatus = null
  while (Date.now() < deadline) {
    if (server.child.exitCode !== null) {
      throw new Error(`${label} exited before becoming ready (last HTTP status: ${lastStatus ?? 'no response'}).\n${server.output}`)
    }
    try {
      const response = await fetch(url)
      lastStatus = response.status
      if (response.ok) return
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Timed out waiting for ${label} (last HTTP status: ${lastStatus ?? 'no response'}).\n${server.output}`)
}

function stopServer(server) {
  if (server?.child.exitCode === null) server.child.kill('SIGTERM')
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const playwrightPackage = JSON.parse(await readFile(path.join(REPOSITORY_ROOT, 'node_modules/playwright/package.json'), 'utf8'))
  if (playwrightPackage.version !== '1.63.0') {
    throw new Error(`Pinned Playwright version 1.63.0 is required; found ${playwrightPackage.version}.`)
  }
  const dataRoot = await realpath(args.dataRoot).catch(async () => {
    await mkdir(args.dataRoot, { recursive: true })
    return realpath(args.dataRoot)
  })
  const profile = path.join(dataRoot, 'browser', 'm0-chromium-profile')
  let cursor = profile
  while (cursor !== dataRoot) {
    try {
      if ((await lstat(cursor)).isSymbolicLink()) throw new Error(`M0 Chromium profile traverses a symbolic link: ${cursor}`)
    } catch (error) {
      if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
        cursor = path.dirname(cursor)
        continue
      }
      throw error
    }
    cursor = path.dirname(cursor)
  }
  await mkdir(path.dirname(profile), { recursive: true, mode: 0o700 })
  const browsers = JSON.parse(await readFile(path.join(REPOSITORY_ROOT, 'node_modules/playwright-core/browsers.json'), 'utf8'))
  const chromiumPin = browsers.browsers?.find((item) => item.name === 'chromium')
  if (!chromiumPin?.revision) throw new Error('Could not read the Playwright-pinned Chromium revision.')
  const executablePath = chromium.executablePath()
  const executable = await lstat(executablePath).catch(() => null)
  if (!executable?.isFile()) {
    throw new Error('Pinned Chromium is not installed. Run `npm run m0:browser:install` once, then retry.')
  }

  const api = start('uv', ['run', '--project', 'services/api', '--locked', 'drawable-api'], {
    DRAWABLE_DATA_ROOT: dataRoot,
    DRAWABLE_API_MODE: 'fixture',
    DRAWABLE_API_HOST: '127.0.0.1',
    DRAWABLE_API_PORT: '8000',
    CURATION_MODE: '1',
  })
  let web
  let context
  try {
    await waitFor(`${API_ORIGIN}/api/v1/health`, api, 'local API')
    web = start(process.execPath, [path.join(REPOSITORY_ROOT, 'node_modules/vite/bin/vite.js'), path.join(REPOSITORY_ROOT, 'apps/web'), '--host', '127.0.0.1', '--port', '5173', '--strictPort'], {
      BROWSER: 'none',
    })
    await waitFor(WEB_ORIGIN, web, 'local Vite app')
    context = await chromium.launchPersistentContext(profile, {
      headless: args.headless,
      executablePath,
      serviceWorkers: 'block',
      args: ['--disable-background-networking', '--disable-component-update', '--disable-sync', '--no-first-run'],
    })
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url())
      const allowed = url.protocol === 'http:'
        && (url.hostname === '127.0.0.1' || url.hostname === 'localhost')
        && ((url.port === '5173') || (url.port === '8000'))
      if (allowed) await route.continue()
      else await route.abort('blockedbyclient')
    })
    const browserVersion = context.browser()?.version() ?? 'unknown'
    const page = context.pages()[0] ?? await context.newPage()
    await page.goto(`${WEB_ORIGIN}/m0`, { waitUntil: 'domcontentloaded' })
    process.stdout.write([
      `M0 owner workspace opened in Drawable's isolated Chromium ${browserVersion}.`,
      `Playwright Chromium revision: ${chromiumPin.revision}.`,
      `Browser profile and accepted recordings: ${path.relative(dataRoot, profile).split(path.sep).join('/')}.`,
      'Network requests are restricted to the local app and API. Close the Chromium window to stop the local servers.',
      '',
    ].join('\n'))
    await new Promise((resolve) => {
      context.once('close', resolve)
      process.once('SIGINT', () => { void context.close(); resolve() })
      process.once('SIGTERM', () => { void context.close(); resolve() })
    })
  } finally {
    if (context) await context.close().catch(() => {})
    stopServer(web)
    stopServer(api)
  }
}

main().catch((error) => {
  process.stderr.write(`M0 local Chromium could not start: ${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 2
})
