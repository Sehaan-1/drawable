import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { DrawingDocument, StoredRasterAsset } from '../lib/types'

/**
 * The resolver contract every surface leans on: a stored import blob that is
 * gone (or undecodable, or behind a storage error) degrades to an `unavailable`
 * entry — never a rejection — and an omission is **not latched**: one transient
 * storage error must not make the artwork permanently absent for canvas,
 * exports, *and* search until reload.
 */

const loadRasterAssetMock = vi.fn<(assetId: string) => Promise<StoredRasterAsset | null>>()

vi.mock('./persistence', () => ({
  loadRasterAsset: (assetId: string) => loadRasterAssetMock(assetId),
}))

/** jsdom has no object URLs and its <img> never loads — stand both in. */
let decodeShouldFail: string[] = []

class FakeImage {
  onload: (() => void) | null = null
  onerror: ((event: unknown) => void) | null = null
  set src(url: string) {
    queueMicrotask(() => {
      if (decodeShouldFail.includes(url)) this.onerror?.(new Error('decode failed'))
      else this.onload?.()
    })
  }
}

vi.stubGlobal('Image', FakeImage)
vi.stubGlobal('URL', Object.assign(URL, {
  createObjectURL: (blob: Blob) => `blob:${blob.size}`,
  revokeObjectURL: () => undefined,
}))

const createImageBitmapMock = vi.fn<(blob: Blob) => Promise<{ blobSize: number }>>()
vi.stubGlobal('createImageBitmap', (blob: Blob) => createImageBitmapMock(blob))

function stored(id: string, content = `blob-of-${id}`): StoredRasterAsset {
  return {
    id,
    mimeType: 'image/png',
    width: 2048,
    height: 2048,
    sha256: '0'.repeat(64),
    blob: new Blob([content], { type: 'image/png' }),
  }
}

function documentWith(ops: Array<{ layer: number; assetId: string; visible?: boolean }>): DrawingDocument {
  const layers = Array.from({ length: 4 }, (_, index) => ({
    id: `layer-${index + 1}`,
    name: `Layer ${index + 1}`,
    visible: true,
    opacity: 1,
    operations: [] as DrawingDocument['layers'][number]['operations'],
  }))
  for (const op of ops) {
    const layer = layers[op.layer]!
    if (op.visible === false) layer.visible = false
    layer.operations.push({
      id: `op-${op.assetId}-${op.layer}`,
      kind: 'raster',
      assetId: op.assetId,
      x: 0,
      y: 0,
      width: 2048,
      height: 2048,
      createdAt: 0,
    })
  }
  return {
    id: 'doc-test',
    title: 'Test',
    revision: 1,
    updatedAt: 0,
    layers,
    trace: { assetId: null, imageUrl: null, visible: false, opacity: 1, scale: 1 },
  }
}

beforeEach(async () => {
  loadRasterAssetMock.mockReset()
  createImageBitmapMock.mockReset().mockImplementation(async (blob) => ({ blobSize: blob.size }))
  decodeShouldFail = []
  const { clearRasterImageCache } = await import('./rasterAssets')
  clearRasterImageCache()
})

async function resolvers() {
  return import('./rasterAssets')
}

describe('resolveRasterImages', () => {
  it('resolves a stored import into the images map', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => stored(id))
    const { resolveRasterImages } = await resolvers()
    const { images, unavailable } = await resolveRasterImages(documentWith([{ layer: 3, assetId: 'raster-a' }]))
    expect(images.get('raster-a')).toBeInstanceOf(FakeImage)
    expect(unavailable).toEqual([])
  })

  it('visible scope ignores artwork only a hidden layer references; all scope includes it', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => (id === 'raster-gone' ? null : stored(id)))
    const document = documentWith([
      { layer: 3, assetId: 'raster-shown' },
      { layer: 2, assetId: 'raster-gone', visible: false },
    ])
    const { resolveRasterImages } = await resolvers()
    const visible = await resolveRasterImages(document, 'visible')
    expect(visible.unavailable).toEqual([])
    expect([...visible.images.keys()]).toEqual(['raster-shown'])
    const all = await resolveRasterImages(document, 'all')
    expect(all.unavailable).toEqual(['raster-gone'])
    expect([...all.images.keys()].sort()).toEqual(['raster-shown'])
  })

  it('reports a gone stored blob as unavailable instead of rejecting, and still resolves the rest', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => (id === 'raster-gone' ? null : stored(id)))
    const { resolveRasterImages } = await resolvers()
    const { images, unavailable } = await resolveRasterImages(documentWith([
      { layer: 3, assetId: 'raster-gone' },
      { layer: 2, assetId: 'raster-kept' },
    ]))
    expect(unavailable).toEqual(['raster-gone'])
    expect(images.has('raster-kept')).toBe(true)
    expect(images.has('raster-gone')).toBe(false)
  })

  it('does not latch a storage error: the next resolve retries and succeeds', async () => {
    // One flaky read — exactly the IndexedDB hiccup the no-latch rule exists for.
    loadRasterAssetMock.mockRejectedValueOnce(new Error('transaction aborted'))
    loadRasterAssetMock.mockImplementation(async (id) => stored(id))
    const { resolveRasterImages } = await resolvers()
    const document = documentWith([{ layer: 3, assetId: 'raster-a' }])
    const first = await resolveRasterImages(document)
    expect(first.unavailable).toEqual(['raster-a'])
    const second = await resolveRasterImages(document)
    expect(second.unavailable).toEqual([])
    expect(second.images.has('raster-a')).toBe(true)
    expect(loadRasterAssetMock).toHaveBeenCalledTimes(2)
  })

  it('reports an undecodable blob as unavailable rather than rejecting', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => stored(id, 'broken'))
    decodeShouldFail = ['blob:6']
    const { resolveRasterImages } = await resolvers()
    const { images, unavailable } = await resolveRasterImages(documentWith([{ layer: 3, assetId: 'raster-a' }]))
    expect(unavailable).toEqual(['raster-a'])
    expect(images.size).toBe(0)
  })

  it('caches a decoded image so concurrent callers share one storage read', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => stored(id))
    const { resolveRasterImages } = await resolvers()
    const document = documentWith([{ layer: 3, assetId: 'raster-a' }, { layer: 2, assetId: 'raster-a' }])
    const first = await resolveRasterImages(document)
    expect(loadRasterAssetMock).toHaveBeenCalledTimes(1)
    const second = await resolveRasterImages(document)
    expect(loadRasterAssetMock).toHaveBeenCalledTimes(1)
    expect(second.images.get('raster-a')).toBe(first.images.get('raster-a'))
  })
})

describe('bitmap and stored-asset resolution', () => {
  it('resolveRasterBitmaps returns surviving bitmaps and names the missing import', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => (id === 'raster-gone' ? null : stored(id)))
    const { resolveRasterBitmaps } = await resolvers()
    const { bitmaps, unavailable } = await resolveRasterBitmaps(documentWith([
      { layer: 3, assetId: 'raster-gone' },
      { layer: 2, assetId: 'raster-kept' },
    ]))
    expect(bitmaps.map((entry) => entry.id)).toEqual(['raster-kept'])
    expect(unavailable).toEqual(['raster-gone'])
  })

  it('loadReferencedRasterAssets returns surviving stored assets and degrades a storage error to an omission', async () => {
    loadRasterAssetMock.mockImplementation(async (id) => {
      if (id === 'raster-error') throw new Error('transaction aborted')
      return stored(id)
    })
    const { loadReferencedRasterAssets } = await resolvers()
    const { assets, unavailable } = await loadReferencedRasterAssets(documentWith([
      { layer: 3, assetId: 'raster-error' },
      { layer: 2, assetId: 'raster-kept' },
    ]))
    expect(assets.map((asset) => asset.id)).toEqual(['raster-kept'])
    expect(unavailable).toEqual(['raster-error'])
  })

  it('summarises omissions for every surface from one sentence', async () => {
    const { unavailableArtworkSummary } = await resolvers()
    expect(unavailableArtworkSummary(['raster-a'])).toBe('One imported artwork is no longer stored on this device')
    expect(unavailableArtworkSummary(['raster-a', 'raster-b'])).toBe('2 imported artworks are no longer stored on this device')
  })
})
