import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { DrawingDocument } from './types'

/**
 * PNG and SVG export degrade: they write exactly what the canvas shows and
 * *return* the ids of the artwork left out, so the dialog can say the file is
 * missing something. A gone stored blob is never a reason to refuse a picture.
 */

const resolveRasterImagesMock = vi.fn()
const loadReferencedRasterAssetsMock = vi.fn()

vi.mock('../services/rasterAssets', () => ({
  resolveRasterImages: (...args: unknown[]) => resolveRasterImagesMock(...args),
  loadReferencedRasterAssets: (...args: unknown[]) => loadReferencedRasterAssetsMock(...args),
}))

const downloaded: Blob[] = []

vi.stubGlobal('URL', Object.assign(URL, {
  createObjectURL: (blob: Blob) => {
    downloaded.push(blob)
    return 'blob:download'
  },
  revokeObjectURL: () => undefined,
}))

interface RecordedContext {
  fillStyle: string
  fills: number
  draws: number
  clearRects: number
  globalAlpha: number
}

const contexts: RecordedContext[] = []

function fakeContext(canvas: HTMLCanvasElement) {
  const record: RecordedContext = { fillStyle: '', fills: 0, draws: 0, clearRects: 0, globalAlpha: 1 }
  contexts.push(record)
  return {
    canvas,
    get fillStyle() { return record.fillStyle },
    set fillStyle(value: string) { record.fillStyle = value },
    get globalAlpha() { return record.globalAlpha },
    set globalAlpha(value: number) { record.globalAlpha = value },
    fillRect: () => { record.fills += 1 },
    drawImage: () => { record.draws += 1 },
    clearRect: () => { record.clearRects += 1 },
  } as unknown as CanvasRenderingContext2D
}

beforeEach(() => {
  downloaded.length = 0
  contexts.length = 0
  resolveRasterImagesMock.mockReset().mockResolvedValue({ images: new Map(), unavailable: [] })
  loadReferencedRasterAssetsMock.mockReset().mockResolvedValue({ assets: [], unavailable: [] })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
  HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement) {
    return fakeContext(this)
  } as unknown as typeof HTMLCanvasElement.prototype.getContext
  HTMLCanvasElement.prototype.toBlob = function (callback) {
    callback(new Blob(['png-bytes'], { type: 'image/png' }))
  }
})

function rasterOp(assetId: string) {
  return { id: `op-${assetId}`, kind: 'raster' as const, assetId, x: 0, y: 0, width: 2048, height: 2048, createdAt: 0 }
}

function documentWithRaster(visibleAsset: string | null, hiddenAsset: string | null): DrawingDocument {
  return {
    id: 'doc-test',
    title: 'Study',
    revision: 1,
    updatedAt: 0,
    layers: Array.from({ length: 4 }, (_, index) => ({
      id: `layer-${index + 1}`,
      name: `Layer ${index + 1}`,
      visible: index === 3 ? true : index === 2 && hiddenAsset !== null ? false : true,
      opacity: 1,
      operations: index === 3 && visibleAsset ? [rasterOp(visibleAsset)] : index === 2 && hiddenAsset ? [rasterOp(hiddenAsset)] : [],
    })),
    trace: { assetId: null, imageUrl: null, visible: false, opacity: 1, scale: 1 },
  }
}

async function blobText(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

describe('PNG export', () => {
  it('writes what the canvas shows and returns the ids of the imports it had to leave out', async () => {
    resolveRasterImagesMock.mockResolvedValue({ images: new Map(), unavailable: ['raster-gone'] })
    const { exportPng } = await import('./exportDocument')
    const unavailable = await exportPng(documentWithRaster('raster-gone', null), false)
    expect(unavailable).toEqual(['raster-gone'])
    expect(downloaded).toHaveLength(1)
    // Opaque export still paints the white ground before the surviving layers.
    expect(contexts[0]?.fillStyle).toBe('#fff')
    expect(contexts[0]?.fills).toBe(1)
  })

  it('returns no omissions when every import resolved, and a transparent export skips the white ground', async () => {
    const { exportPng } = await import('./exportDocument')
    const unavailable = await exportPng(documentWithRaster('raster-kept', null), true)
    expect(unavailable).toEqual([])
    expect(downloaded).toHaveLength(1)
    expect(contexts[0]?.fills).toBe(0)
  })

  it('asks only for visible-scope artwork, so a hidden layer’s gap is not reported as a PNG omission', async () => {
    const { exportPng } = await import('./exportDocument')
    await exportPng(documentWithRaster('raster-kept', 'raster-hidden'), false)
    expect(resolveRasterImagesMock).toHaveBeenCalledWith(expect.anything(), 'visible')
  })
})

describe('SVG export', () => {
  it('embeds the surviving artwork, skips the missing one silently in markup, and reports it to the caller', async () => {
    loadReferencedRasterAssetsMock.mockResolvedValue({
      assets: [{ id: 'raster-kept', mimeType: 'image/png', width: 2048, height: 2048, sha256: 'a'.repeat(64), blob: new Blob(['kept'], { type: 'image/png' }) }],
      unavailable: ['raster-gone'],
    })
    const { exportSvg } = await import('./exportDocument')
    const document = documentWithRaster('raster-kept', null)
    document.layers[2]!.operations.push(rasterOp('raster-gone'))
    const unavailable = await exportSvg(document)
    expect(unavailable).toEqual(['raster-gone'])
    const svg = await blobText(downloaded[0]!)
    // The surviving asset is embedded; nothing dangles for the missing one.
    expect(svg).toContain('<image href="data:image/png;base64,')
    expect(svg).not.toContain('raster-gone')
    expect(svg.match(/<image /g)).toHaveLength(1)
  })
})
