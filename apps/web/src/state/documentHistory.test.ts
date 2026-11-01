import { describe, expect, it } from 'vitest'
import type { DrawingDocument, DrawingOperation, StrokeOperation } from '../lib/types'
import {
  applyHistoryEntry,
  createHistoryEntry,
  DEFAULT_HISTORY_LIMITS,
  estimateOperationBytes,
  findLayer,
  trimFuture,
  trimPast,
  type HistoryEntry,
  type HistoryPayload,
} from './documentHistory'

function stroke(id: string, points = 3): StrokeOperation {
  return {
    id,
    kind: 'stroke',
    tool: 'pressure',
    points: Array.from({ length: points }, (_, index) => ({ x: index, y: index, pressure: 0.5, time: index })),
    size: 12,
    smoothing: 0.1,
    streamline: 0,
    simulatePressure: false,
    createdAt: 0,
  }
}

function makeDocument(): DrawingDocument {
  return {
    id: 'document-1',
    title: 'Untitled drawing',
    revision: 0,
    updatedAt: 0,
    layers: [1, 2, 3, 4].map((index) => ({
      id: `layer-${index}`,
      name: `Layer ${index}`,
      visible: true,
      opacity: 1,
      operations: [] as DrawingOperation[],
    })),
    trace: { assetId: null, imageUrl: null, visible: true, opacity: 0.3, scale: 1 },
  }
}

function entry(payload: HistoryPayload): HistoryEntry {
  return createHistoryEntry(payload)
}

describe('document history helpers', () => {
  it('estimates stroke payload by point count and raster payload as constant', () => {
    expect(estimateOperationBytes(stroke('a', 10))).toBe(96 + 10 * 48)
    expect(estimateOperationBytes({
      id: 'raster-1', kind: 'raster', assetId: 'asset-1', x: 0, y: 0, width: 10, height: 10, createdAt: 0,
    })).toBe(96 + 64)
  })

  it('weights each entry kind', () => {
    expect(entry({ kind: 'operation_appended', layerId: 'layer-1', operation: stroke('a', 2) }).bytes).toBe(96 + 2 * 48)
    expect(entry({ kind: 'layer_cleared', layerId: 'layer-1', operations: [stroke('a', 1), stroke('b', 2)] }).bytes).toBe((96 + 48) + (96 + 96))
    expect(entry({ kind: 'layer_moved', layerId: 'layer-1', from: 0, to: 1 }).bytes).toBe(128)
    expect(entry({ kind: 'layer_properties_changed', layerId: 'layer-1', before: { name: 'Ink', visible: true, opacity: 1 }, after: { name: 'Inks', visible: true, opacity: 1 } }).bytes).toBe(128 + 2 * (3 + 4))
    expect(entry({ kind: 'title_changed', before: 'abc', after: 'abcd' }).bytes).toBe(64 + 2 * (3 + 4))
    expect(entry({ kind: 'trace_changed', before: { assetId: 'a', imageUrl: 'bb' }, after: { assetId: null, imageUrl: null } }).bytes).toBe(128 + 2 * (2 + 1))
  })

  it('trims past by entry count, keeping the newest entries', () => {
    const entries = [stroke('a'), stroke('b'), stroke('c')].map((operation) => entry({ kind: 'operation_appended', layerId: 'layer-1', operation }))
    const trimmed = trimPast(entries, { maxEntries: 2, maxBytes: Number.MAX_SAFE_INTEGER })
    expect(trimmed.map((item) => item.kind === 'operation_appended' && item.operation.id)).toEqual(['b', 'c'])
    expect(trimmed[1]).toBe(entries[2])
  })

  it('trims past by byte budget but always keeps the newest entry', () => {
    const entries = [stroke('a'), stroke('b')].map((operation) => entry({ kind: 'operation_appended', layerId: 'layer-1', operation }))
    expect(trimPast(entries, { maxEntries: 500, maxBytes: 200 }).map((item) => item.kind === 'operation_appended' && item.operation.id)).toEqual(['b'])
    expect(trimPast(entries, { maxEntries: 500, maxBytes: 10 })).toHaveLength(1)
    expect(trimPast(entries, DEFAULT_HISTORY_LIMITS)).toBe(entries)
  })

  it('trims future from the oldest end, keeping the next redo entry', () => {
    const entries = [stroke('a'), stroke('b'), stroke('c')].map((operation) => entry({ kind: 'operation_appended', layerId: 'layer-1', operation }))
    const trimmed = trimFuture(entries, { maxEntries: 2, maxBytes: Number.MAX_SAFE_INTEGER })
    expect(trimmed).toHaveLength(2)
    expect(trimmed[0]).toBe(entries[0])
    expect(trimFuture(entries, DEFAULT_HISTORY_LIMITS)).toBe(entries)
  })

  it('applies operation_appended without mutating the input document', () => {
    const document = makeDocument()
    const operation = stroke('a')
    const appended = applyHistoryEntry(document, entry({ kind: 'operation_appended', layerId: 'layer-1', operation }), 'redo')
    expect(findLayer(appended, 'layer-1')?.operations).toEqual([operation])
    expect(findLayer(document, 'layer-1')?.operations).toEqual([])
    expect(appended.layers[1]).toBe(document.layers[1])
    expect(applyHistoryEntry(appended, entry({ kind: 'operation_appended', layerId: 'layer-1', operation }), 'undo').layers[0]?.operations).toEqual([])
    // Idempotent redo.
    const redoEntry = entry({ kind: 'operation_appended', layerId: 'layer-1', operation })
    expect(applyHistoryEntry(appended, redoEntry, 'redo')).toBe(appended)
  })

  it('applies layer_properties_changed, layer_moved, layer_cleared, title and trace entries', () => {
    const document = { ...makeDocument(), title: 'Old', trace: { assetId: 'a', imageUrl: 'url', visible: true, opacity: 0.5, scale: 1 } }

    const changed = applyHistoryEntry(document, entry({
      kind: 'layer_properties_changed', layerId: 'layer-2',
      before: { name: 'Layer 2', visible: true, opacity: 1 },
      after: { name: 'Inks', visible: false, opacity: 0.4 },
    }), 'redo')
    expect(changed.layers[1]).toMatchObject({ name: 'Inks', visible: false, opacity: 0.4 })
    expect(changed.layers[0]).toBe(document.layers[0])
    expect(applyHistoryEntry(changed, entry({
      kind: 'layer_properties_changed', layerId: 'layer-2',
      before: { name: 'Layer 2', visible: true, opacity: 1 },
      after: { name: 'Inks', visible: false, opacity: 0.4 },
    }), 'undo').layers[1]).toMatchObject({ name: 'Layer 2', visible: true, opacity: 1 })

    const moved = applyHistoryEntry(document, entry({ kind: 'layer_moved', layerId: 'layer-1', from: 0, to: 2 }), 'redo')
    expect(moved.layers.map((layer) => layer.id)).toEqual(['layer-2', 'layer-3', 'layer-1', 'layer-4'])
    expect(applyHistoryEntry(moved, entry({ kind: 'layer_moved', layerId: 'layer-1', from: 0, to: 2 }), 'undo').layers.map((layer) => layer.id))
      .toEqual(['layer-1', 'layer-2', 'layer-3', 'layer-4'])

    const operations = [stroke('a'), stroke('b')]
    const seeded = { ...document, layers: document.layers.map((layer) => (layer.id === 'layer-3' ? { ...layer, operations } : layer)) }
    const cleared = applyHistoryEntry(seeded, entry({ kind: 'layer_cleared', layerId: 'layer-3', operations }), 'redo')
    expect(findLayer(cleared, 'layer-3')?.operations).toEqual([])
    expect(findLayer(applyHistoryEntry(cleared, entry({ kind: 'layer_cleared', layerId: 'layer-3', operations }), 'undo'), 'layer-3')?.operations).toBe(operations)

    expect(applyHistoryEntry(document, entry({ kind: 'title_changed', before: 'Old', after: 'New' }), 'redo').title).toBe('New')
    expect(applyHistoryEntry(document, entry({ kind: 'title_changed', before: 'Old', after: 'New' }), 'undo').title).toBe('Old')

    const traceEntry = entry({ kind: 'trace_changed', before: { visible: true }, after: { visible: false } })
    expect(applyHistoryEntry(document, traceEntry, 'redo').trace).toEqual({ ...document.trace, visible: false })
    expect(applyHistoryEntry(document, traceEntry, 'undo').trace).toEqual(document.trace)
  })
})
