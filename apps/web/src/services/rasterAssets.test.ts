import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DrawingDocument, DrawingOperation, StrokeOperation } from '../lib/types'

const assets = vi.hoisted(() => new Map<string, { id: string; blob: Blob; createdAt: number }>())

vi.mock('./persistence', () => ({
  loadRasterAsset: vi.fn(async (assetId: string) => assets.get(assetId) ?? null),
}))

const { referencedAssetIds, resolveRasterBitmaps } = await import('./rasterAssets')

function stroke(id: string): StrokeOperation {
  return {
    id, kind: 'stroke', tool: 'pressure', points: [{ x: 0, y: 0, pressure: 0.5, time: 0 }],
    size: 8, smoothing: 0.55, streamline: 0.45, simulatePressure: false, createdAt: 0,
  }
}

function makeDocument(layers: DrawingOperation[][]): DrawingDocument {
  return {
    id: 'document-1',
    title: 'Untitled drawing',
    revision: 0,
    updatedAt: 0,
    layers: layers.map((operations, index) => ({
      id: `layer-${index + 1}`,
      name: `Layer ${index + 1}`,
      visible: true,
      opacity: 1,
      operations,
    })),
    trace: { assetId: null, imageUrl: null, visible: true, opacity: 0.3, scale: 1 },
  }
}

afterEach(() => {
  vi.restoreAllMocks()
  assets.clear()
  delete (globalThis as { createImageBitmap?: unknown }).createImageBitmap
})

describe('raster asset lookup', () => {
  it('reports no asset IDs for a pure-vector document and caches per operations array', () => {
    const flatMap = vi.spyOn(Array.prototype, 'flatMap')
    const document = makeDocument([[], [], [], []])
    expect(referencedAssetIds(document)).toEqual([])
    expect(flatMap).toHaveBeenCalledTimes(4)

    // A commit replaces only layer-1's operations array; the other three are cache hits.
    flatMap.mockClear()
    const committed = makeDocument([[stroke('a')], document.layers[1]!.operations, document.layers[2]!.operations, document.layers[3]!.operations])
    expect(referencedAssetIds(committed)).toEqual([])
    expect(flatMap).toHaveBeenCalledTimes(1)

    // Same arrays again → no rescans at all.
    flatMap.mockClear()
    expect(referencedAssetIds(committed)).toEqual([])
    expect(flatMap).toHaveBeenCalledTimes(0)
  })

  it('resolves raster operations and rejects unknown assets', async () => {
    assets.set('asset-1', { id: 'asset-1', blob: new Blob(['image']), createdAt: 0 })
    const bitmap = { width: 1, height: 1 }
    ;(globalThis as { createImageBitmap?: unknown }).createImageBitmap = vi.fn(async () => bitmap)

    const document = makeDocument([[], [
      { id: 'raster-1', kind: 'raster', assetId: 'asset-1', x: 0, y: 0, width: 10, height: 10, createdAt: 0 },
    ], [], []])
    expect(referencedAssetIds(document)).toEqual(['asset-1'])
    expect(await resolveRasterBitmaps(document)).toEqual([{ id: 'asset-1', bitmap }])

    const missing = makeDocument([[], [
      { id: 'raster-2', kind: 'raster', assetId: 'missing', x: 0, y: 0, width: 10, height: 10, createdAt: 0 },
    ], [], []])
    await expect(resolveRasterBitmaps(missing)).rejects.toThrow('Imported artwork missing is unavailable.')
  })
})
