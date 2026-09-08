import { rasterizeDocument } from '../lib/exportDocument'
import { measureInk, type RasterInk } from '../lib/rasterInk'
import type { DrawingDocument } from '../lib/types'
import type { SearchOwnership } from '../state/searchStore'
import { resolveRasterBitmaps } from './rasterAssets'

/**
 * A prepared snapshot carries the *whole* request identity it was made for, so
 * the caller can gate on ownership instead of comparing a subset of fields:
 * a snapshot that does not echo `token`+document+revision+generation belongs to
 * a request that has already lost the results panel.
 *
 * `ink` is the measurement the sufficiency rule is based on (see
 * `lib/rasterInk.ts`); `null` means the environment gave us no readable pixels,
 * which is deliberately *not* the same as "blank".
 */
export interface PreparedSnapshot {
  token: number
  documentId: string
  revision: number
  generation: number
  image: Blob
  ink: RasterInk | null
  worker: boolean
}

interface PendingSnapshot {
  resolve: (value: PreparedSnapshot) => void
  reject: (reason: unknown) => void
}

let worker: Worker | null = null
const pending = new Map<string, PendingSnapshot>()

/** Identity fields echoed by the worker alongside the rendered snapshot. */
interface SnapshotIdentity {
  token: number
  documentId: string
  revision: number
  generation: number
}

function isSnapshotMessage(data: unknown): data is SnapshotIdentity & { id: string; image?: Blob; ink?: RasterInk | null; error?: string } {
  return typeof data === 'object' && data !== null && typeof (data as { id?: unknown }).id === 'string'
}

function snapshotWorker() {
  if (worker) return worker
  worker = new Worker(new URL('./snapshot.worker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = (event: MessageEvent<unknown>) => {
    const data = event.data
    if (!isSnapshotMessage(data)) return
    const request = pending.get(data.id)
    if (!request) return
    pending.delete(data.id)
    if (data.error || !data.image) request.reject(new Error(data.error ?? 'Snapshot worker returned no image'))
    else request.resolve({ token: data.token, documentId: data.documentId, revision: data.revision, generation: data.generation, image: data.image, ink: data.ink ?? null, worker: true })
  }
  worker.onerror = (event) => {
    for (const request of pending.values()) request.reject(new Error(event.message))
    pending.clear()
    worker?.terminate()
    worker = null
  }
  return worker
}

function canvasBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('Canvas snapshot failed')), 'image/png'))
}

function contextInk(context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null, width: number, height: number): RasterInk | null {
  // No 2D surface (jsdom, or a canvas that failed to allocate) means no
  // measurement — never a verdict that the drawing is empty.
  if (!context || typeof context.getImageData !== 'function') return null
  try {
    const pixels = context.getImageData(0, 0, width, height)
    return measureInk(pixels.data, pixels.width, pixels.height)
  } catch {
    return null
  }
}

const SNAPSHOT_SIZE = 512

async function mainThreadFallback(document: DrawingDocument, identity: SnapshotIdentity): Promise<PreparedSnapshot> {
  const source = await rasterizeDocument(document, false)
  const target = globalThis.document.createElement('canvas')
  target.width = SNAPSHOT_SIZE
  target.height = SNAPSHOT_SIZE
  const context = target.getContext('2d')
  context?.drawImage(source, 0, 0, SNAPSHOT_SIZE, SNAPSHOT_SIZE)
  return { ...identity, image: await canvasBlob(target), ink: contextInk(context, SNAPSHOT_SIZE, SNAPSHOT_SIZE), worker: false }
}

/** Tell the worker to drop a request so a cancelled snapshot stops costing work. */
function cancelSnapshot(id: string) {
  worker?.postMessage({ cancel: id })
}

/**
 * Render the 512×512 search snapshot for exactly one request identity.
 *
 * The returned promise either resolves with that identity echoed back, or
 * rejects with an `AbortError` when the caller's signal fires — including while
 * bitmaps are still being decoded for transfer, where the transferred objects
 * are closed here so an abort cannot leak them.
 */
export async function prepareSnapshot(document: DrawingDocument, ownership: SearchOwnership, signal: AbortSignal): Promise<PreparedSnapshot> {
  const identity: SnapshotIdentity = {
    token: ownership.token,
    documentId: document.id,
    revision: document.revision,
    generation: ownership.generation,
  }
  if (signal.aborted) throw new DOMException('Snapshot cancelled', 'AbortError')
  if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined') return mainThreadFallback(document, identity)
  const rasterAssets = await resolveRasterBitmaps(document)
  if (signal.aborted) {
    for (const asset of rasterAssets) asset.bitmap.close()
    throw new DOMException('Snapshot cancelled', 'AbortError')
  }
  const id = crypto.randomUUID()
  return new Promise<PreparedSnapshot>((resolve, reject) => {
    const onAbort = () => {
      const request = pending.get(id)
      if (!request) return
      pending.delete(id)
      // The bitmaps are already transferred (neutered) on the worker side, so
      // the worker is told to close them and skip the render entirely.
      cancelSnapshot(id)
      request.reject(new DOMException('Snapshot cancelled', 'AbortError'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
    pending.set(id, {
      resolve: (value) => { signal.removeEventListener('abort', onAbort); resolve(value) },
      reject: (error) => { signal.removeEventListener('abort', onAbort); reject(error) },
    })
    snapshotWorker().postMessage({ id, ...identity, document, rasterAssets }, rasterAssets.map((asset) => asset.bitmap))
  })
}
