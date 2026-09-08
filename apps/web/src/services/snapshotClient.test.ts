import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { RasterInk } from '../lib/rasterInk'
import type { DrawingDocument } from '../lib/types'
import type { SearchOwnership } from '../state/searchStore'

/**
 * The snapshot client is the first half of request ownership: it renders for one
 * identity and echoes that identity back, so the caller can refuse a snapshot
 * that no longer belongs to the request it is answering. Aborts have to be
 * honoured at both ends — before the worker is even asked, and while bitmaps are
 * still being decoded for transfer — and an abandoned snapshot must not leak the
 * transferred objects or leave a spinner behind.
 */

const INK: RasterInk = { inkPixels: 4096, coverage: 4096 / (512 * 512), bboxDiagonalRatio: 0.62 }

const document_: DrawingDocument = {
  id: 'doc-a',
  title: 'Untitled drawing',
  revision: 7,
  updatedAt: 0,
  layers: [],
  trace: { assetId: null, imageUrl: null, visible: false, opacity: 1, scale: 1 },
}

const ownership = (over: Partial<SearchOwnership> = {}): SearchOwnership => ({
  token: 42,
  documentId: document_.id,
  revision: document_.revision,
  generation: 3,
  ...over,
})

interface Reply {
  id: string
  token: number
  documentId: string
  revision: number
  generation: number
  image?: Blob
  ink?: RasterInk | null
  error?: string
}

class FakeWorker {
  static instances: FakeWorker[] = []
  onmessage: ((event: { data: unknown }) => void) | null = null
  onerror: ((event: { message: string }) => void) | null = null
  messages: Array<{ data: unknown; transfer: unknown[] }> = []
  terminated = false

  constructor() {
    FakeWorker.instances.push(this)
  }

  postMessage(data: unknown, transfer?: unknown[]) {
    this.messages.push({ data, transfer: transfer ?? [] })
  }

  terminate() {
    this.terminated = true
  }

  get render() {
    return this.messages.find((message) => !(message.data as { cancel?: string }).cancel)?.data as
      | Record<string, unknown>
      | undefined
  }

  get cancelId() {
    return (this.messages.find((message) => (message.data as { cancel?: string }).cancel)?.data as { cancel: string })
      ?.cancel
  }

  reply(data: Reply) {
    this.onmessage?.({ data })
  }

  fail(message: string) {
    this.onerror?.({ message })
  }
}

function fakeBitmap() {
  return { close: vi.fn() } as unknown as ImageBitmap & { close: ReturnType<typeof vi.fn> }
}

/** Identity fields the worker is expected to carry through untouched. */
function echoOf(data: Record<string, unknown>, id: string): Reply {
  return {
    id,
    token: data.token as number,
    documentId: data.documentId as string,
    revision: data.revision as number,
    generation: data.generation as number,
    image: new Blob(['png'], { type: 'image/png' }),
    ink: INK,
  }
}

const resolveRasterBitmapsMock = vi.fn<() => Promise<Array<{ id: string; bitmap: ImageBitmap }>>>()
const rasterizeMock = vi.fn<() => Promise<CanvasImageSource>>()

vi.mock('./rasterAssets', () => ({
  resolveRasterBitmaps: () => resolveRasterBitmapsMock(),
  // `rasterizeDocument` (the fallback's raster source) resolves imports here too.
  resolveRasterImages: () => rasterizeImagesMock(),
}))
vi.mock('../lib/exportDocument', () => ({
  rasterizeDocument: () => rasterizeMock(),
}))
const rasterizeImagesMock = vi.fn<() => Promise<Map<string, CanvasImageSource>>>()

beforeEach(() => {
  FakeWorker.instances = []
  resolveRasterBitmapsMock.mockReset().mockResolvedValue([])
  rasterizeMock.mockReset().mockImplementation(async () => ({ width: 2048, height: 2048 }) as unknown as CanvasImageSource)
  rasterizeImagesMock.mockReset().mockResolvedValue(new Map())
  vi.resetModules()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function loadClient(withWorker = true) {
  if (withWorker) {
    vi.stubGlobal('Worker', FakeWorker as unknown as typeof Worker)
    vi.stubGlobal('OffscreenCanvas', class {} as unknown as typeof OffscreenCanvas)
  } else {
    vi.stubGlobal('Worker', undefined)
    vi.stubGlobal('OffscreenCanvas', undefined)
    // jsdom gives a canvas no 2D surface, which is exactly the "unmeasurable"
    // case the fallback has to survive; it has no encoder either, so only the
    // blob hand-off is stood in for.
    HTMLCanvasElement.prototype.getContext = (() => null) as unknown as typeof HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.toBlob = function (callback) {
      callback(new Blob(['png'], { type: 'image/png' }))
    }
  }
  const { prepareSnapshot } = await import('./snapshotClient')
  const worker = () => FakeWorker.instances[FakeWorker.instances.length - 1]!
  return { prepareSnapshot, worker }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve
    reject = innerReject
  })
  promise.catch(() => undefined)
  return { promise, resolve, reject }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

describe('snapshot identity', () => {
  it('echoes the whole ownership token back to the caller', async () => {
    const { prepareSnapshot, worker } = await loadClient()
    const pending = prepareSnapshot(document_, ownership(), new AbortController().signal)
    await flush()
    const instance = worker()
    const message = instance.render!
    const id = message.id as string
    expect(message).toMatchObject({ token: 42, documentId: 'doc-a', revision: 7, generation: 3 })
    instance.reply(echoOf(message, id))
    await expect(pending).resolves.toMatchObject({
      token: 42,
      documentId: 'doc-a',
      revision: 7,
      generation: 3,
      ink: INK,
      worker: true,
    })
  })

  it('reports a snapshot that answers a different generation as such', async () => {
    // The worker is a separate thread: it can legitimately answer after the
    // identity moved on. It must not silently look like a fresh answer.
    const { prepareSnapshot, worker } = await loadClient()
    const pending = prepareSnapshot(document_, ownership(), new AbortController().signal)
    await flush()
    const instance = worker()
    const message = instance.render!
    instance.reply({ ...echoOf(message, message.id as string), generation: 2 })
    const snapshot = await pending
    expect(snapshot.generation).toBe(2)
    expect(snapshot.generation).not.toBe(ownership().generation)
  })

  it('keeps a reply for a request it no longer tracks from breaking the client', async () => {
    const { prepareSnapshot, worker } = await loadClient()
    const pending = prepareSnapshot(document_, ownership(), new AbortController().signal)
    await flush()
    const instance = worker()
    const message = instance.render!
    const id = message.id as string
    instance.reply({ id: 'untracked', token: 1, documentId: 'x', revision: 1, generation: 1 })
    instance.reply(echoOf(message, id))
    await expect(pending).resolves.toMatchObject({ token: 42 })
  })

  it('rejects every in-flight request when the worker dies, and drops it', async () => {
    const { prepareSnapshot, worker } = await loadClient()
    const first = prepareSnapshot(document_, ownership({ token: 1 }), new AbortController().signal)
    await flush()
    const second = prepareSnapshot(document_, ownership({ token: 2, generation: 4 }), new AbortController().signal)
    await flush()
    const instance = worker()
    instance.fail('snapshot canvas failed')
    await expect(first).rejects.toThrow('snapshot canvas failed')
    await expect(second).rejects.toThrow('snapshot canvas failed')
    expect(instance.terminated).toBe(true)
  })
})

describe('snapshot cancellation', () => {
  it('never asks the worker for a request that is already aborted', async () => {
    const { prepareSnapshot, worker } = await loadClient()
    const controller = new AbortController()
    controller.abort()
    await expect(prepareSnapshot(document_, ownership(), controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    })
    expect(FakeWorker.instances).toHaveLength(0)
  })

  it('tells the worker to drop a snapshot aborted while it was rendering', async () => {
    const { prepareSnapshot, worker } = await loadClient()
    const controller = new AbortController()
    const pending = prepareSnapshot(document_, ownership(), controller.signal)
    await flush()
    const instance = worker()
    const id = instance.render!.id as string
    controller.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    expect(instance.cancelId).toBe(id)
    // The late answer belongs to nothing now: it must not resolve the request.
    instance.reply(echoOf(instance.render!, id))
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('closes bitmaps when the abort lands during decoding', async () => {
    const gate = deferred<void>()
    const assets = [
      { id: 'asset-a', bitmap: fakeBitmap() },
      { id: 'asset-b', bitmap: fakeBitmap() },
    ]
    resolveRasterBitmapsMock.mockImplementation(async () => {
      await gate.promise
      return assets
    })
    const { prepareSnapshot } = await loadClient()
    const controller = new AbortController()
    const pending = prepareSnapshot(document_, ownership(), controller.signal)
    controller.abort()
    gate.resolve()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    expect(assets.every((asset) => asset.bitmap.close.mock.calls.length === 1)).toBe(true)
    expect(FakeWorker.instances).toHaveLength(0)
  })

  it('transfers the decoded bitmaps to the worker for rendering', async () => {
    const bitmap = fakeBitmap()
    resolveRasterBitmapsMock.mockResolvedValue([{ id: 'asset-a', bitmap }])
    const { prepareSnapshot, worker } = await loadClient()
    const pending = prepareSnapshot({ ...document_, revision: 9 }, ownership(), new AbortController().signal)
    await flush()
    const instance = worker()
    // The bitmaps are transferred (neutered on this side) rather than cloned.
    expect(instance.messages[0]?.transfer).toEqual([bitmap])
    const message = instance.render!
    expect(message.rasterAssets).toEqual([{ id: 'asset-a', bitmap }])
    instance.reply(echoOf(message, message.id as string))
    await expect(pending).resolves.toMatchObject({ revision: 9 })
  })
})

describe('snapshot measurement without a 2D surface', () => {
  it('measures nothing rather than guessing "blank"', async () => {
    const { prepareSnapshot } = await loadClient(false)
    const snapshot = await prepareSnapshot({ ...document_, revision: 9 }, ownership({ revision: 9 }), new AbortController().signal)
    expect(snapshot.worker).toBe(false)
    expect(snapshot.ink).toBeNull()
    // `revision` is the document's own, not the caller's copy of it.
    expect(snapshot).toMatchObject({ token: 42, documentId: 'doc-a', revision: 9, generation: 3 })
  })
})
