import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { DrawingDocument, StoredRasterAsset } from './types'
import { exportDrawableProject, MissingProjectArtworkError, prepareImport } from './projectFiles'

const rasterMocks = vi.hoisted(() => ({ loadReferencedRasterAssets: vi.fn() }))

vi.mock('../services/rasterAssets', async () => {
  const actual = await vi.importActual<typeof import('../services/rasterAssets')>('../services/rasterAssets')
  return {
    ...actual,
    loadReferencedRasterAssets: (...args: unknown[]) => rasterMocks.loadReferencedRasterAssets(...args),
  }
})

const downloads: Blob[] = []

vi.stubGlobal('URL', Object.assign(URL, {
  createObjectURL: (blob: Blob) => {
    downloads.push(blob)
    return 'blob:download'
  },
  revokeObjectURL: () => undefined,
}))

function project(overrides: Record<string, unknown> = {}) {
  return {
    format: 'drawable-project',
    formatVersion: 1,
    applicationVersion: '0.1.0',
    exportedAt: new Date(0).toISOString(),
    activeLayerId: 'layer-1',
    assets: [],
    document: {
      title: 'Round trip',
      layers: Array.from({ length: 4 }, (_, index) => ({
        id: `layer-${index + 1}`,
        name: `Layer ${index + 1}`,
        visible: true,
        opacity: 1,
        operations: [],
      })),
      trace: { assetId: 'fixture-1', visible: true, opacity: 0.3, scale: 1 },
    },
    ...overrides,
  }
}

function projectFile(value: unknown) {
  const source = JSON.stringify(value)
  const file = new File([source], 'study.drawable', { type: 'application/vnd.drawable.project+json' })
  Object.defineProperty(file, 'text', { value: async () => source })
  return file
}

describe('drawable project import', () => {
  it('validates four editable layers while keeping trace metadata portable', async () => {
    const imported = await prepareImport(projectFile(project()))
    expect(imported.sourceKind).toBe('project')
    expect(imported.document.layers).toHaveLength(4)
    expect(imported.document.trace).toMatchObject({ assetId: 'fixture-1', imageUrl: null, opacity: 0.3 })
    expect(imported.activeLayerId).toBe('layer-1')
  })

  it('rejects newer project versions without returning partial data', async () => {
    await expect(prepareImport(projectFile(project({ formatVersion: 2 })))).rejects.toThrow('newer version')
  })

  it('rejects documents that violate the fixed four-layer contract', async () => {
    const invalid = project()
    invalid.document.layers.pop()
    await expect(prepareImport(projectFile(invalid))).rejects.toThrow('exactly four layers')
  })
})

describe('drawable project export with a stored import blob gone', () => {
  function rasterOp(assetId: string) {
    return { id: `op-${assetId}`, kind: 'raster' as const, assetId, x: 0, y: 0, width: 2048, height: 2048, createdAt: 0 }
  }

  /** A document whose visible ink uses `keptId` and whose hidden layer uses `goneId`. */
  async function documentWith(keptId: string, goneId: string): Promise<DrawingDocument> {
    return {
      id: 'doc-export',
      title: 'Study',
      revision: 3,
      updatedAt: 0,
      layers: Array.from({ length: 4 }, (_, index) => ({
        id: `layer-${index + 1}`,
        name: `Layer ${index + 1}`,
        visible: index !== 2,
        opacity: 1,
        operations: index === 3 ? [rasterOp(keptId)] : index === 2 ? [rasterOp(goneId)] : [],
      })),
      trace: { assetId: null, imageUrl: null, visible: false, opacity: 1, scale: 1 },
    }
  }

  async function storedAsset(content: string): Promise<StoredRasterAsset> {
    const blob = new Blob([content], { type: 'image/png' })
    const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())
    const checksum = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
    return { id: `raster-${checksum}`, mimeType: 'image/png', width: 2048, height: 2048, sha256: checksum, blob }
  }

  async function downloadedProject() {
    expect(downloads).toHaveLength(1)
    const text = await downloads[0]!.text()
    return JSON.parse(text) as Record<string, any>
  }

  beforeEach(() => {
    downloads.length = 0
    rasterMocks.loadReferencedRasterAssets.mockReset()
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
  })

  it('refuses to write an editable file that could not be reopened, and names the missing ids', async () => {
    const kept = await storedAsset('kept-png')
    rasterMocks.loadReferencedRasterAssets.mockResolvedValue({ assets: [kept], unavailable: ['raster-gone-a', 'raster-gone-b'] })
    const document = await documentWith(kept.id, 'raster-gone-a')
    const failure = await exportDrawableProject(document, 'layer-1').catch((error) => error)
    expect(failure).toBeInstanceOf(MissingProjectArtworkError)
    expect(failure.unavailable).toEqual(['raster-gone-a', 'raster-gone-b'])
    expect(failure.message).toContain('raster-gone-a')
    expect(failure.message).toContain('raster-gone-b')
    expect(downloads).toHaveLength(0)
  })

  it('drops both sides on explicit omission — dead ops and orphaned blobs — and the file must pass the importer back', async () => {
    const kept = await storedAsset('kept-png')
    rasterMocks.loadReferencedRasterAssets.mockResolvedValue({ assets: [kept], unavailable: ['raster-gone'] })
    const document = await documentWith(kept.id, 'raster-gone')
    const result = await exportDrawableProject(document, 'layer-1', { omitUnavailable: true })
    expect(result.omitted).toEqual(['raster-gone'])

    const written = await downloadedProject()
    const operations = (written.document as any).layers.flatMap((layer: any) => layer.operations)
    // Neither the reference nor the blob survives: a dangling reference *and*
    // an unreferenced asset are both import rejections.
    expect(operations.some((op: any) => op.assetId === 'raster-gone')).toBe(false)
    expect(operations.some((op: any) => op.assetId === kept.id)).toBe(true)
    expect(written.assets).toHaveLength(1)
    expect(written.assets[0].id).toBe(kept.id)

    // The export already asserted this internally; prove it once more exactly
    // the way a user would see it — handing the file to the import dialog.
    const roundTrip = await prepareImport(projectFile(written))
    expect(roundTrip.sourceKind).toBe('project')
    expect(roundTrip.assets).toHaveLength(1)
  })

  it('exports a drawing with all artwork present exactly as before', async () => {
    const kept = await storedAsset('kept-png')
    rasterMocks.loadReferencedRasterAssets.mockResolvedValue({ assets: [kept], unavailable: [] })
    const document = await documentWith(kept.id, 'unused-hidden')
    document.layers[2]!.operations = []
    const result = await exportDrawableProject(document, 'layer-1')
    expect(result).toEqual({ omitted: [] })
    const written = await downloadedProject()
    expect((written.document as any).layers[3].operations).toHaveLength(1)
    expect(written.assets).toHaveLength(1)
    expect(written.activeLayerId).toBe('layer-1')
  })
})
