import { createHash } from 'node:crypto'
import { spawn, spawnSync } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, lstat, realpath, writeFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { chromium } from 'playwright'

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, '..')
const DEFAULT_DATA_ROOT = path.join(REPOSITORY_ROOT, 'datasets')
const DEFAULT_ORIGIN = 'http://127.0.0.1:4175'
const MAX_BATCH_BYTES = 250 * 1024 * 1024
const REQUIRED_RECORDINGS = 10

function parseArgs(argv) {
  const result = {
    dataRoot: DEFAULT_DATA_ROOT,
    batch: null,
    report: 'reports/private/m0-recording-replay-v1.json',
    gateReport: 'reports/m0-ten-feasibility-sequences-v1.json',
    recordingIds: [],
  }
  for (let index = 0; index < argv.length; index += 1) {
    const option = argv[index]
    if (option === '--recording-id') {
      result.recordingIds.push(argv[index + 1])
      index += 1
    } else if (option === '--data-root') {
      result.dataRoot = path.resolve(REPOSITORY_ROOT, argv[index + 1])
      index += 1
    } else if (option === '--batch') {
      result.batch = argv[index + 1]
      index += 1
    } else if (option === '--report') {
      result.report = argv[index + 1]
      index += 1
    } else if (option === '--gate-report') {
      result.gateReport = argv[index + 1]
      index += 1
    } else if (option === '--help' || option === '-h') {
      result.help = true
    } else {
      throw new Error(`Unknown option: ${option}`)
    }
  }
  return result
}

function printHelp() {
  process.stdout.write([
    'Independent M0 recording replay using the Playwright-pinned Chromium binary.',
    '',
    'Usage: npm run m0:replay -- --data-root datasets --batch reports/private/recording-batches/<sha>.json',
    '  --recording-id <id> (repeat exactly ten times)',
    '  [--report reports/private/m0-recording-replay-v1.json]',
    '  [--gate-report reports/m0-ten-feasibility-sequences-v1.json]',
    '',
  ].join('\n'))
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

async function sha256File(file) {
  const hash = createHash('sha256')
  for await (const block of createReadStream(file)) hash.update(block)
  return hash.digest('hex')
}

function privatePath(root, relative) {
  if (typeof relative !== 'string' || !relative || path.isAbsolute(relative) || relative.includes('\\')) {
    throw new Error('Evidence paths must be nonempty relative POSIX paths inside --data-root.')
  }
  const parts = relative.split('/')
  if (parts.some((part) => !part || part === '.' || part === '..')) {
    throw new Error('Evidence paths may not contain empty, current, or parent path components.')
  }
  const target = path.join(root, ...parts)
  if (target !== root && !target.startsWith(`${root}${path.sep}`)) {
    throw new Error('Evidence path escaped --data-root.')
  }
  return target
}

async function rejectSymlinkPath(root, target) {
  let cursor = target
  while (cursor !== root) {
    try {
      if ((await lstat(cursor)).isSymbolicLink()) throw new Error(`Symbolic link is not allowed in evidence path: ${cursor}`)
    } catch (error) {
      if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
        cursor = path.dirname(cursor)
        continue
      }
      throw error
    }
    cursor = path.dirname(cursor)
  }
}

async function immutableJson(root, relative, value) {
  const target = privatePath(root, relative)
  await rejectSymlinkPath(root, target)
  await mkdir(path.dirname(target), { recursive: true })
  const body = `${JSON.stringify(value, null, 2)}\n`
  try {
    await writeFile(target, body, { encoding: 'utf8', flag: 'wx', mode: 0o600 })
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST') {
      throw new Error(`Evidence reports are immutable; choose a new path: ${relative}`)
    }
    throw error
  }
  return { path: relative, sha256: sha256(Buffer.from(body, 'utf8')) }
}

async function fileFingerprint(relative) {
  const file = path.join(REPOSITORY_ROOT, relative)
  return { path: relative, sha256: await sha256File(file) }
}

function gitValue(args) {
  const result = spawnSync('git', args, { cwd: REPOSITORY_ROOT, encoding: 'utf8' })
  return result.status === 0 ? result.stdout.trim() : null
}

async function startVite() {
  const viteCli = path.join(REPOSITORY_ROOT, 'node_modules', 'vite', 'bin', 'vite.js')
  const server = spawn(process.execPath, [viteCli, path.join(REPOSITORY_ROOT, 'apps/web'), '--host', '127.0.0.1', '--port', '4175', '--strictPort'], {
    cwd: REPOSITORY_ROOT,
    env: { ...process.env, BROWSER: 'none' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let logs = ''
  const collect = (chunk) => { logs = `${logs}${chunk.toString()}`.slice(-6000) }
  server.stdout.on('data', collect)
  server.stderr.on('data', collect)
  const deadline = Date.now() + 30_000
  let lastStatus = null
  while (Date.now() < deadline) {
    if (server.exitCode !== null) {
      throw new Error(`Vite exited before becoming ready (last HTTP status: ${lastStatus ?? 'no response'}).\n${logs}`)
    }
    try {
      const response = await fetch(DEFAULT_ORIGIN)
      lastStatus = response.status
      if (response.ok) return { server, logs }
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  server.kill('SIGTERM')
  throw new Error(`Timed out waiting for the local Vite server (last HTTP status: ${lastStatus ?? 'no response'}).\n${logs}`)
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    printHelp()
    return 0
  }
  const dataRoot = await realpath(args.dataRoot).catch(async () => {
    await mkdir(args.dataRoot, { recursive: true })
    return realpath(args.dataRoot)
  })
  if (!args.batch) throw new Error('--batch is required and must name an imported, private JSON batch.')
  if (args.recordingIds.length !== REQUIRED_RECORDINGS || args.recordingIds.some((id) => !id)) {
    throw new Error('Select exactly ten distinct --recording-id values from the accepted recording batch.')
  }
  if (new Set(args.recordingIds).size !== REQUIRED_RECORDINGS) {
    throw new Error('The ten selected recording IDs must be unique.')
  }

  const batchPath = privatePath(dataRoot, args.batch)
  await rejectSymlinkPath(dataRoot, batchPath)
  const batchStat = await lstat(batchPath)
  if (!batchStat.isFile() || batchStat.size > MAX_BATCH_BYTES) {
    throw new Error('Recording batch must be a regular file no larger than 250 MiB.')
  }
  const batchBytes = await readFile(batchPath)
  const batchSha256 = sha256(batchBytes)
  let batch
  try { batch = JSON.parse(batchBytes.toString('utf8')) }
  catch (error) { throw new Error(`Recording batch is not valid UTF-8 JSON: ${error}`) }
  if (batch?.format !== 'drawable-sequence-batch' || batch?.formatVersion !== 1 || !Array.isArray(batch.records)) {
    throw new Error('Recording batch format or version is unsupported.')
  }
  const records = new Map()
  for (const entry of batch.records) {
    if (entry?.status !== 'accepted' || !entry.sequence?.recordingId || records.has(entry.sequence.recordingId)) continue
    records.set(entry.sequence.recordingId, entry.sequence)
  }
  const selected = args.recordingIds.map((id) => {
    const sequence = records.get(id)
    if (!sequence) throw new Error(`Selected recording is missing or not accepted in the batch: ${id}`)
    if (sequence.formatVersion !== 3 || sequence.migrationReview) {
      throw new Error(`M0 replay requires current version 3 records without legacy migration review: ${id}`)
    }
    if (sequence.reference?.kind !== 'none') {
      throw new Error(`M0 feasibility recordings must be freehand with no reference: ${id}`)
    }
    return sequence
  })

  const packageLockPath = path.join(REPOSITORY_ROOT, 'package-lock.json')
  const playwrightPackage = JSON.parse(await readFile(path.join(REPOSITORY_ROOT, 'node_modules/playwright/package.json'), 'utf8'))
  if (playwrightPackage.version !== '1.63.0') {
    throw new Error(`Pinned Playwright version 1.63.0 is required; found ${playwrightPackage.version}.`)
  }
  const browsersFile = path.join(REPOSITORY_ROOT, 'node_modules/playwright-core/browsers.json')
  const browserPins = JSON.parse(await readFile(browsersFile, 'utf8'))
  const chromiumPin = browserPins.browsers?.find((browser) => browser.name === 'chromium')
  if (!chromiumPin?.revision) throw new Error('Could not read the Playwright-pinned Chromium revision.')
  const executablePath = chromium.executablePath()
  const executableStat = await lstat(executablePath).catch(() => null)
  if (!executableStat?.isFile()) {
    throw new Error('Pinned Chromium is not installed. Run `npm run m0:browser:install` once, then retry.')
  }

  const sourcePaths = [
    'apps/web/src/services/recordingValidation.ts',
    'apps/web/src/services/renderChecksum.ts',
    'apps/web/src/services/recordingTypes.ts',
    'apps/web/src/lib/exportDocument.ts',
    'apps/web/src/lib/drawing.ts',
    'apps/web/src/lib/types.ts',
    'apps/web/src/state/documentStore.ts',
  ]
  const implementationFingerprints = await Promise.all(sourcePaths.map(fileFingerprint))
  const codeRevision = gitValue(['rev-parse', 'HEAD'])
  const workingTreeStatus = gitValue(['status', '--short'])
  const lockSha256 = await sha256File(packageLockPath)
  const executableSha256 = await sha256File(executablePath)
  const { server } = await startVite()
  let browser
  let browserVersion = null
  const startedAt = new Date().toISOString()
  let outcomes = []
  let runtimeError = null
  try {
    browser = await chromium.launch({ headless: true })
    browserVersion = browser.version()
    const context = await browser.newContext({ serviceWorkers: 'block' })
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url())
      if (url.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(url.hostname) && url.port === '4175') {
        await route.continue()
      } else {
        await route.abort('blockedbyclient')
      }
    })
    const page = await context.newPage()
    const pageErrors = []
    page.on('pageerror', (error) => pageErrors.push(error.message))
    await page.goto(`${DEFAULT_ORIGIN}/record`, { waitUntil: 'domcontentloaded' })
    for (const sequence of selected) {
      const result = await page.evaluate(async (input) => {
        const validation = await import('/src/services/recordingValidation.ts')
        const checksum = await import('/src/services/renderChecksum.ts')
        const validationResult = validation.validateAndReplay(input)
        if (!validationResult.valid) return { valid: false, errors: validationResult.errors }
        const replayed = validation.replaySequence(input)
        const renderedSha256 = await checksum.checksumRenderedDocument(replayed)
        return {
          valid: true,
          renderedSha256,
          expectedRenderedSha256: input.finalRenderedSha256,
          strokeCount: input.statistics.strokeCount,
          operationCount: input.statistics.operationCount,
          pointCount: input.statistics.pointCount,
          elapsedMs: input.elapsedMs,
          acceptedAt: input.acceptedAt,
          consent: {
            creatorConfirmed: input.consent.creatorConfirmed,
            contributorCanGrantPermissions: input.consent.contributorCanGrantPermissions,
            research: input.consent.research,
            evaluation: input.consent.evaluation,
            referenceRightsDeclared: input.consent.referenceRightsDeclared,
          },
          hasExternalReference: input.reference.kind !== 'none',
        }
      }, sequence)
      outcomes.push({
        recording_id: sequence.recordingId,
        status: result.valid && result.renderedSha256 === sequence.finalRenderedSha256 ? 'verified' : 'blocked',
        ...result,
      })
    }
    if (pageErrors.length) runtimeError = `Browser module errors: ${pageErrors.join('; ')}`
    await context.close()
  } catch (error) {
    runtimeError = error instanceof Error ? error.message : String(error)
  } finally {
    if (browser) await browser.close().catch(() => {})
    server.kill('SIGTERM')
  }

  const blockers = []
  if (runtimeError) blockers.push(runtimeError)
  if (outcomes.length !== REQUIRED_RECORDINGS) blockers.push(`Only ${outcomes.length} of ten recordings replayed.`)
  for (const outcome of outcomes) {
    if (outcome.status !== 'verified') blockers.push(`Recording failed independent replay or render hash verification: ${outcome.recording_id}`)
    if (!outcome.consent?.creatorConfirmed || !outcome.consent?.contributorCanGrantPermissions || !outcome.consent?.research || !outcome.consent?.evaluation || !outcome.consent?.referenceRightsDeclared) {
      blockers.push(`Required research and evaluation consent is absent: ${outcome.recording_id}`)
    }
    if (outcome.hasExternalReference) blockers.push(`Feasibility recording contains a reference: ${outcome.recording_id}`)
  }
  const replayArtifact = {
    schema_version: 1,
    artifact_type: 'drawable-m0-recording-replay',
    status: blockers.length ? 'blocked' : 'verified',
    created_at: new Date().toISOString(),
    replay_started_at: startedAt,
    selected_recording_ids: args.recordingIds,
    batch_path: args.batch,
    batch_sha256: batchSha256,
    browser: {
      playwright_version: playwrightPackage.version,
      chromium_revision: chromiumPin.revision,
      chromium_version: browserVersion,
      executable_sha256: executableSha256,
      headless: true,
      external_requests_blocked: true,
    },
    implementation: {
      git_revision: codeRevision,
      working_tree_status: workingTreeStatus,
      package_lock_sha256: lockSha256,
      source_files: implementationFingerprints,
    },
    outcomes,
    blockers,
    training_started: false,
    data_uploaded: false,
  }
  const replayWritten = await immutableJson(dataRoot, args.report, replayArtifact)
  if (blockers.length) {
    process.stderr.write(`M0 replay blocked. Diagnostic report: ${replayWritten.path}\n${blockers.join('\n')}\n`)
    return 2
  }

  const gateReport = {
    schema_version: 2,
    requirements_revision: '2026-10-04-practical-r1',
    artifact_type: 'drawable-gate-report',
    gate_id: 'ten_feasibility_sequences',
    producer: 'tools/replay-m0-recordings.mjs',
    produced_at: new Date().toISOString(),
    checks: { consent_rechecked: true, replay_verified: true, render_hashes_verified: true },
    measurements: {
      accepted_sequences: outcomes.length,
      median_elapsed_seconds: (() => {
        const sorted = outcomes.map((row) => row.elapsedMs / 1000).sort((a, b) => a - b)
        return (sorted[4] + sorted[5]) / 2
      })(),
      total_strokes: outcomes.reduce((sum, row) => sum + (row.strokeCount ?? 0), 0),
    },
    inputs: [
      { path: args.batch, sha256: batchSha256 },
      replayWritten,
    ],
    summary: `Ten distinct consented freehand sequences were independently replayed in Playwright-pinned Chromium revision ${chromiumPin.revision}; every canonical RGBA render hash matched.`,
  }
  const gateWritten = await immutableJson(dataRoot, args.gateReport, gateReport)
  process.stdout.write(`${JSON.stringify({ status: 'complete', replay_report: replayWritten, gate_report: gateWritten, accepted_sequences: outcomes.length }, null, 2)}\n`)
  return 0
}

main().then((code) => { process.exitCode = code }).catch((error) => {
  process.stderr.write(`M0 recording replay failed: ${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 2
})
