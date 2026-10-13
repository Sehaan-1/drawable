import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'

const root = path.resolve(import.meta.dirname, '..')
const origins = ['http://127.0.0.1:5173', 'http://127.0.0.1:8000']
const pause = (ms) => new Promise(resolve => setTimeout(resolve, ms))
async function reachable(origin) {
  try { return (await fetch(origin + (origin.endsWith('8000') ? '/api/v1/health' : '/'), {signal: AbortSignal.timeout(1000)})).ok }
  catch { return false }
}
if ((await Promise.all(origins.map(reachable))).some(Boolean)) throw new Error('Close the running app before this isolated startup/shutdown check.')

const child = spawn(process.execPath, ['tools/review-local.mjs', '--search', '--headless'], {cwd: root, stdio: ['ignore', 'pipe', 'pipe']})
let output = ''
for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => { output = (output + chunk.toString()).slice(-6000) })
const exited = new Promise(resolve => child.once('exit', resolve))
let browser
const errors = []
const requests = []
let report
try {
  const deadline = Date.now() + 90000
  while (!(await reachable(origins[0]))) {
    if (child.exitCode !== null || Date.now() >= deadline) throw new Error('Search launcher did not become ready.\n' + output)
    await pause(250)
  }
  const health = await (await fetch(origins[1] + '/api/v1/health')).json()
  if (!health.ready || !health.models_loaded || health.mode !== 'cpu' || !health.gallery_size) throw new Error('The launcher did not load a real CPU gallery.')
  browser = await chromium.launch({headless: true})
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}})
  await context.route('**/*', route => origins.includes(new URL(route.request().url()).origin) ? route.continue() : route.abort())
  const page = await context.newPage()
  page.on('pageerror', error => errors.push(error.message))
  page.on('response', async response => {
    if (response.url().endsWith('/api/v1/search') && response.ok()) requests.push(await response.json())
  })
  await page.goto(origins[0] + '/draw?mode=real')
  const canvas = page.getByLabel('Drawing canvas')
  await canvas.waitFor({state: 'visible'})
  const box = await canvas.boundingBox()
  if (!box) throw new Error('The drawing canvas is unavailable.')
  const strokes = [
    [[.40,.30],[.46,.26],[.53,.29],[.55,.36],[.52,.43],[.45,.45],[.39,.39],[.40,.30]],
    [[.45,.45],[.44,.53],[.35,.57],[.31,.73]],
    [[.52,.43],[.53,.53],[.62,.57],[.67,.72]],
    [[.44,.53],[.49,.57],[.53,.53]],
  ]
  for (const stroke of strokes) {
    await page.mouse.move(box.x + box.width * stroke[0][0], box.y + box.height * stroke[0][1])
    await page.mouse.down()
    for (const [x, y] of stroke.slice(1)) await page.mouse.move(box.x + box.width * x, box.y + box.height * y, {steps: 6})
    await page.mouse.up()
  }
  await page.locator('.reference-card img').first().waitFor({state: 'visible', timeout: 20000})
  await page.waitForFunction(() => [...document.querySelectorAll('.reference-card img')].every(img => img.complete && img.naturalWidth > 0))
  const images = await page.locator('.reference-card img').evaluateAll(items => items.map(img => img.getAttribute('src')))
  const response = requests.findLast(value => value.mode === 'provisional')
  if (errors.length || !images.length || !images.every(url => url.startsWith('/api/v1/assets/')) || !response?.build_id || response.training_started !== false || response.data_uploaded !== false) throw new Error('The browser did not return intact real references.')
  const preferredStyle = response.groups.find(group => group.id.startsWith('style_'))?.title
  let stylePreferenceVerified = false
  if (preferredStyle) {
    const styleChip = page.getByRole('button', {name: preferredStyle.split(' ')[0], exact: true})
    await styleChip.click()
    await page.waitForFunction(style => {
      const titles = [...document.querySelectorAll('.reference-section h3')].map(node => node.textContent?.trim())
      return titles[0] === 'Best Match' && titles[1] === style
    }, preferredStyle)
    const orderedGroups = await page.locator('.reference-section h3').allTextContents()
    if (orderedGroups[0] !== 'Best Match' || orderedGroups[1] !== preferredStyle) throw new Error(`The preferred style did not move to the top of the real search groups: ${JSON.stringify(orderedGroups)}.`)
    stylePreferenceVerified = true
  }
  await page.getByLabel('Pin reference').first().click()
  await page.getByRole('heading', {name: 'Pinned', exact: true}).waitFor({state: 'visible'})
  const persistedPinCount = await page.evaluate(() => JSON.parse(localStorage.getItem('drawable-local-pins') ?? '[]').length)
  if (!persistedPinCount) throw new Error('Pinning a real reference did not persist locally.')
  await page.goto(origins[0] + '/draw?mode=real')
  const restoredCanvas = await page.getByLabel('Drawing canvas').boundingBox()
  if (!restoredCanvas) throw new Error('The drawing canvas disappeared after opening a new workspace.')
  for (const stroke of strokes) {
    await page.mouse.move(restoredCanvas.x + restoredCanvas.width * stroke[0][0], restoredCanvas.y + restoredCanvas.height * stroke[0][1])
    await page.mouse.down()
    for (const [x, y] of stroke.slice(1)) await page.mouse.move(restoredCanvas.x + restoredCanvas.width * x, restoredCanvas.y + restoredCanvas.height * y, {steps: 6})
    await page.mouse.up()
  }
  await page.getByRole('heading', {name: 'Pinned', exact: true}).waitFor({state: 'visible', timeout: 20000})
  if (await page.evaluate(() => JSON.parse(localStorage.getItem('drawable-local-pins') ?? '[]').length) !== persistedPinCount) throw new Error('A saved local pin did not survive workspace recovery.')
  const healthDuringSearch = await (await fetch(origins[1] + '/api/v1/health')).json()
  if (!healthDuringSearch.ready) throw new Error('Search made the health handler unavailable.')
  const warmSearchSamples = requests.filter(value => value.mode === 'provisional').slice(-4).map(value => value.timings_ms.total_ms)
  const sortedSamples = [...warmSearchSamples].sort((left, right) => left - right)
  const middle = Math.floor(sortedSamples.length / 2)
  const medianSearchMs = !sortedSamples.length ? null : sortedSamples.length % 2 ? sortedSamples[middle] : (sortedSamples[middle - 1] + sortedSamples[middle]) / 2
  report = {schema_version: 1, kind: 'prototype-browser-smoke', build_id: response.build_id, gallery_size: health.gallery_size, cards_loaded: images.length, last_search_ms: response.timings_ms.total_ms, warm_search_samples_ms: warmSearchSamples, warm_search_median_ms: medianSearchMs, warm_search_max_ms: sortedSamples.at(-1) ?? null, style_preference_verified: stylePreferenceVerified, pins_persisted_across_workspace: true, browser_errors: errors, input: 'synthetic-head-and-shoulders', retrieval_quality_evaluated: false, training_started: false, data_uploaded: false}
} finally {
  if (browser) await browser.close()
  child.kill('SIGTERM')
  await Promise.race([exited, pause(10000).then(() => { throw new Error('The launcher did not stop its child processes.') })])
}
const shutdownDeadline = Date.now() + 5000
while ((await Promise.all(origins.map(reachable))).some(Boolean)) {
  if (Date.now() > shutdownDeadline) throw new Error('A local server remained running after launcher shutdown.')
  await pause(100)
}
report.shutdown_verified = true
const directory = path.join(root, 'datasets/reports/private/prototype-search-checks')
await mkdir(directory, {recursive: true, mode: 0o700})
const filename = path.join(directory, Date.now() + '.json')
await writeFile(filename, JSON.stringify(report, null, 2) + '\n', {flag: 'wx', mode: 0o600})
process.stdout.write(JSON.stringify({...report, report_path: filename}) + '\n')
