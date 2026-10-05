import { spawn, spawnSync } from 'node:child_process'
import { chmod, lstat, mkdir, readFile, realpath } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { chromium } from 'playwright'

const ROOT = path.resolve(import.meta.dirname, '..')
const WEB_ORIGIN = 'http://127.0.0.1:5173'
const API_ORIGIN = 'http://127.0.0.1:8000'

function parseArgs(argv) {
  let dataRoot = path.join(ROOT, 'datasets')
  let headless = false
  let guided = false
  let search = false
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--data-root') {
      if (!argv[index + 1]) throw new Error('--data-root needs a directory path')
      dataRoot = path.resolve(ROOT, argv[index + 1])
      index += 1
    } else if (argv[index] === '--headless') {
      headless = true
    } else if (argv[index] === '--guided') {
      guided = true
    } else if (argv[index] === '--search') {
      search = true
    } else if (argv[index] === '--help' || argv[index] === '-h') {
      process.stdout.write('Usage: npm run review -- [--data-root datasets] [--headless] [--guided] [--search]\nDefault: local educational image review. --guided opens source/model terms setup. --search opens prepared local search.\n')
      process.exit(0)
    } else {
      throw new Error(`Unknown option: ${argv[index]}`)
    }
  }
  if (guided && search) throw new Error('--guided and --search cannot be combined.')
  return { dataRoot, headless, guided, search }
}

function start(command, args, env) {
  const child = spawn(command, args, {
    cwd: ROOT,
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
  const deadline = Date.now() + 60_000
  let lastStatus = null
  while (Date.now() < deadline) {
    if (server.child.exitCode !== null) {
      throw new Error(`${label} exited before it became ready (last HTTP status: ${lastStatus ?? 'no response'}).\n${server.output}`)
    }
    try {
      const response = await fetch(url)
      lastStatus = response.status
      if (response.ok) return
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 300))
  }
  throw new Error(`Timed out waiting for ${label} (last HTTP status: ${lastStatus ?? 'no response'}).\n${server.output}`)
}

function stop(server) {
  if (server?.child.exitCode === null) server.child.kill('SIGTERM')
}

async function allowedTermsPages() {
  const [sources, models] = await Promise.all([
    readFile(path.join(ROOT, 'config/sources.json'), 'utf8').then(JSON.parse),
    readFile(path.join(ROOT, 'config/models.json'), 'utf8').then(JSON.parse),
  ])
  return [
    ...sources.sources.map((source) => source.terms_url || source.authority_url),
    ...models.models.map((model) => model.terms_source_url).filter(Boolean),
  ].map((value) => {
    const url = new URL(value)
    return { host: url.hostname, path: url.pathname }
  })
}

async function installPinnedBrowser(executablePath) {
  const executable = await lstat(executablePath).catch(() => null)
  if (executable?.isFile()) return
  process.stdout.write('Installing the Chromium version pinned by this project…\n')
  const result = spawnSync(process.execPath, [
    path.join(ROOT, 'node_modules/playwright/cli.js'), 'install', 'chromium',
  ], { cwd: ROOT, stdio: 'inherit' })
  if (result.status !== 0) throw new Error('Pinned Chromium did not install. Check the network connection and run `npm run m0:browser:install`.')
}

async function ensurePrivateBrowserProfile(dataRoot, browserProfile) {
  let cursor = path.dirname(browserProfile)
  while (cursor !== dataRoot) {
    const relative = path.relative(dataRoot, cursor)
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      throw new Error('The Chromium review profile must stay inside the private data root.')
    }
    const existing = await lstat(cursor).catch((error) => {
      if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return null
      throw error
    })
    if (existing?.isSymbolicLink()) throw new Error(`The Chromium review profile traverses a symbolic link: ${cursor}`)
    cursor = path.dirname(cursor)
  }
  await mkdir(path.dirname(browserProfile), { recursive: true, mode: 0o700 })
  await chmod(path.dirname(browserProfile), 0o700)
  const existing = await lstat(browserProfile).catch((error) => {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return null
    throw error
  })
  if (existing && (!existing.isDirectory() || existing.isSymbolicLink())) {
    throw new Error('The Chromium review profile path must be a regular private directory.')
  }
  if (!existing) await mkdir(browserProfile, { mode: 0o700 })
  await chmod(browserProfile, 0o700)
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const pinned = JSON.parse(await readFile(path.join(ROOT, 'node_modules/playwright/package.json'), 'utf8'))
  if (pinned.version !== '1.63.0') throw new Error(`Playwright 1.63.0 is pinned by the workspace; found ${pinned.version}.`)
  const dataRoot = await realpath(options.dataRoot).catch(async () => {
    await mkdir(options.dataRoot, { recursive: true, mode: 0o700 })
    return realpath(options.dataRoot)
  })
  const browserProfile = path.join(dataRoot, 'browser', 'review-chromium-profile')
  await ensurePrivateBrowserProfile(dataRoot, browserProfile)
  const browsers = JSON.parse(await readFile(path.join(ROOT, 'node_modules/playwright-core/browsers.json'), 'utf8'))
  const chromiumPin = browsers.browsers?.find((item) => item.name === 'chromium')
  if (!chromiumPin?.revision) throw new Error('Could not read Playwright’s pinned Chromium revision.')
  const executablePath = chromium.executablePath()
  await installPinnedBrowser(executablePath)

  const api = start('uv', ['run', '--project', 'services/api', '--locked', 'drawable-api'], {
    DRAWABLE_DATA_ROOT: dataRoot,
    DRAWABLE_API_MODE: options.search ? 'real' : 'fixture',
    DRAWABLE_API_HOST: '127.0.0.1',
    DRAWABLE_API_PORT: '8000',
    CURATION_MODE: options.search ? '0' : '1',
    DRAWABLE_LOCAL_PROTOTYPE: options.guided ? '0' : '1',
    HF_HUB_OFFLINE: '1',
    TRANSFORMERS_OFFLINE: '1',
    CUDA_VISIBLE_DEVICES: '',
  })
  let web
  let context
  try {
    await waitFor(`${API_ORIGIN}/api/v1/health`, api, 'the local review API')
    if (options.search) {
      const response = await fetch(`${API_ORIGIN}/api/v1/ready`)
      if (!response.ok) throw new Error('Local search is not prepared. Run npm run search:prepare, then npm run search:local.\n' + api.output)
    }
    web = start(process.execPath, [
      path.join(ROOT, 'node_modules/vite/bin/vite.js'), path.join(ROOT, 'apps/web'), '--host', '127.0.0.1', '--port', '5173', '--strictPort',
    ], { BROWSER: 'none', VITE_DRAWABLE_SEARCH_MODE: options.search ? 'real' : 'fixture' })
    await waitFor(WEB_ORIGIN, web, 'the local web app')
    context = await chromium.launchPersistentContext(browserProfile, {
      headless: options.headless,
      executablePath,
      serviceWorkers: 'block',
      args: ['--disable-background-networking', '--disable-component-update', '--disable-sync', '--no-first-run'],
    })
    const permittedTerms = await allowedTermsPages()
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url())
      const local = url.protocol === 'http:'
        && ['127.0.0.1', 'localhost'].includes(url.hostname)
        && ['5173', '8000'].includes(url.port)
      const officialTerms = !options.search && url.protocol === 'https:'
        && permittedTerms.some((page) => url.hostname === page.host && url.pathname.startsWith(page.path))
      if (local || officialTerms) await route.continue()
      else await route.abort('blockedbyclient')
    })
    const page = context.pages()[0] ?? await context.newPage()
    const reviewPath = options.search ? '/draw?mode=real' : options.guided ? '/setup' : '/curate?prototype=1'
    await page.goto(`${WEB_ORIGIN}${reviewPath}`, { waitUntil: 'domcontentloaded' })
    process.stdout.write([
      `Private local review workspace is open at ${WEB_ORIGIN}${reviewPath}.`,
      options.search ? 'Frozen local CPU search over your saved references. Draw a head or figure and pause for tentative matches.' : options.guided ? 'Guided source/model terms review.' : 'Local educational prototype: review real downloaded sketches directly.',
      `Pinned Chromium revision: ${chromiumPin.revision}.`,
      options.search ? 'Search runs locally; external browser requests are blocked.' : 'Only this app, its local API, and listed official terms pages may use the network.',
      options.search ? 'Close this browser window or press Ctrl+C to stop the app, API, and model worker.' : 'Keep this window open while a source archive is being prepared. Close it to stop both local servers.',
      '',
    ].join('\n'))
    await new Promise((resolve) => {
      context.once('close', resolve)
      process.once('SIGINT', () => { void context.close(); resolve() })
      process.once('SIGTERM', () => { void context.close(); resolve() })
    })
  } finally {
    if (context) await context.close().catch(() => {})
    stop(web)
    stop(api)
  }
}

main().catch((error) => {
  process.stderr.write(`Local review workspace could not start: ${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 2
})
