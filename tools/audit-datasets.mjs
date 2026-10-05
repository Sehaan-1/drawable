#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { stat, readdir, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const rootArgIndex = process.argv.indexOf('--root')
const root = resolve(rootArgIndex >= 0 ? process.argv[rootArgIndex + 1] ?? 'datasets' : 'datasets')
const writeIndex = process.argv.indexOf('--write')
const outputPath = writeIndex >= 0 ? resolve(process.argv[writeIndex + 1] ?? join(root, 'intake-manifest.json')) : null

const sources = [
  { id: 'anita-dataset', title: 'AnitaDataset', authority: 'https://github.com/zhenglinpan/AnitaDataset', download: 'https://drive.google.com/file/d/1ctfD0sMpT2pVutJUOlyEYKhAxufMYmZ_/view?usp=sharing', expectedNames: ['AnitaDataset.zip'], partialPattern: /^AnitaDataset.*\.zip\.part$/i, minimumBytes: 7_000_000_000, rights: 'Per-work terms in the official repository; verify each work before gallery use.' },
  { id: 'emoreccom-panels', title: 'EmoRecCom panels', authority: 'https://l3i-share.univ-lr.fr/2024visemocomic/', download: 'https://l3i-share.univ-lr.fr/2024visemocomic/emoreccom_panels.zip', expectedNames: ['emoreccom_panels.zip'], partialPattern: /^emoreccom_panels.*\.zip\.part$/i, minimumBytes: 191_545_672, rights: 'Public-domain panels according to the official dataset page; retain attribution and source records.' },
  { id: 'emoreccom-annotations', title: 'EmoRecCom annotations', authority: 'https://l3i-share.univ-lr.fr/2024visemocomic/', download: 'https://l3i-share.univ-lr.fr/2024visemocomic/visemocomic_csvs.zip', expectedNames: ['visemocomic_csvs.zip'], partialPattern: /^visemocomic_csvs.*\.zip\.part$/i, minimumBytes: 438_469, rights: 'Use only rows/images covered by the official EmoRecCom terms; referenced external datasets remain separate.' },
]

async function fileInfo(path) {
  try { const details = await stat(path); return { path, bytes: details.size } } catch { return null }
}

async function sha256(path) {
  const hash = createHash('sha256')
  await new Promise((resolvePromise, reject) => {
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('end', resolvePromise)
    stream.on('error', reject)
  })
  return hash.digest('hex')
}

async function archiveTest(path) {
  try { await execFileAsync('unzip', ['-t', path], { maxBuffer: 1024 * 1024 }); return { available: true, valid: true } } catch (error) {
    return { available: error?.code !== 'ENOENT', valid: false, error: error?.code === 'ENOENT' ? 'unzip is unavailable' : 'archive test failed' }
  }
}

const names = await readdir(root).catch(() => [])
const report = { schemaVersion: 1, auditedAt: new Date().toISOString(), root, policy: 'This audit never extracts, renames, deletes, or uploads source archives. Incomplete or unverified archives are not eligible for ingestion.', sources: [] }
for (const source of sources) {
  const exactFiles = (await Promise.all(source.expectedNames.map((name) => fileInfo(join(root, name))))).filter(Boolean)
  const exact = exactFiles.find((candidate) => candidate.bytes > 0) ?? exactFiles[0]
  const partialName = names.find((name) => source.partialPattern.test(name))
  const partial = partialName ? await fileInfo(join(root, partialName)) : null
  const candidate = (partial && partial.bytes > 0) ? partial : exact
  const outputSource = { id: source.id, title: source.title, authority: source.authority, download: source.download, rights: source.rights, minimumBytes: source.minimumBytes, partialPattern: source.partialPattern.source }
  if (!candidate) { report.sources.push({ ...outputSource, status: 'missing' }); continue }
  if (candidate.bytes === 0) { report.sources.push({ ...outputSource, status: 'empty', file: candidate.path, bytes: 0 }); continue }
  if (candidate.path.endsWith('.part') || candidate.bytes < source.minimumBytes) { report.sources.push({ ...outputSource, status: 'partial', file: candidate.path, bytes: candidate.bytes }); continue }
  const archive = await archiveTest(candidate.path)
  if (!archive.valid) { report.sources.push({ ...outputSource, status: 'invalid', file: candidate.path, bytes: candidate.bytes, archive }); continue }
  report.sources.push({ ...outputSource, status: 'complete', file: candidate.path, bytes: candidate.bytes, sha256: await sha256(candidate.path), archive })
}
const output = `${JSON.stringify(report, null, 2)}\n`
if (outputPath) await writeFile(outputPath, output, 'utf8')
process.stdout.write(output)
