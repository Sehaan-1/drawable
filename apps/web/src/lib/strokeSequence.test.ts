import { describe, expect, it } from 'vitest'
import { buildStrokeSequence, documentCounts } from './strokeSequence'
import { LOGICAL_SIZE, type DrawingLayer, type RasterOperation, type StrokeOperation } from './types'

/**
 * The counts on the wire are validated element-wise against the delivered
 * payload (`stroke_count_mismatch` is a 422, not a warning), so payload and
 * counts must come from one source. These are the two cases that used to
 * disagree: an import, which is content without vector geometry, and ink on a
 * layer that is hidden or fully transparent, which is neither.
 */

function stroke(id: string, points: number, tool: StrokeOperation['tool'] = 'pressure'): StrokeOperation {
  return {
    id,
    kind: 'stroke',
    tool,
    points: Array.from({ length: points }, (_, index) => ({
      x: index * 4,
      y: index * 3,
      pressure: 0.5,
      time: index * 16,
    })),
    size: 6,
    smoothing: 0.4,
    streamline: 0.3,
    simulatePressure: false,
    createdAt: 0,
  }
}

function raster(id: string): RasterOperation {
  return { id, kind: 'raster', assetId: `asset-${id}`, x: 0, y: 0, width: 512, height: 512, createdAt: 0 }
}

function layer(id: string, over: Partial<DrawingLayer> = {}): DrawingLayer {
  return { id, name: id, visible: true, opacity: 1, operations: [], ...over }
}

const pointCount = (sequence: ReturnType<typeof buildStrokeSequence>) =>
  sequence.strokes.reduce((total, item) => total + item.points.length, 0)

describe('documentCounts', () => {
  it('reports a raster-only import as content with no vector geometry', () => {
    const layers = [layer('import', { operations: [raster('r1')] })]
    const counts = documentCounts(layers)
    expect(counts).toEqual({ strokeCount: 0, pointCount: 0, rasterCount: 1 })
    // Nothing to send on the stroke branch: an empty sequence, not a fake one.
    expect(buildStrokeSequence(layers).strokes).toEqual([])
  })

  it('counts several imported operations without touching the stroke counts', () => {
    const layers = [layer('import', { operations: [raster('r1'), raster('r2'), stroke('s1', 5)] })]
    expect(documentCounts(layers)).toEqual({ strokeCount: 1, pointCount: 5, rasterCount: 2 })
  })

  it('excludes hidden and fully transparent layers from both halves', () => {
    const layers = [
      layer('visible', { operations: [stroke('s1', 3)] }),
      layer('hidden', { visible: false, operations: [stroke('s2', 9), raster('r1')] }),
      layer('transparent', { opacity: 0, operations: [stroke('s3', 7)] }),
    ]
    const counts = documentCounts(layers)
    const sequence = buildStrokeSequence(layers)
    expect(counts).toEqual({ strokeCount: 1, pointCount: 3, rasterCount: 0 })
    expect(sequence.strokes).toHaveLength(counts.strokeCount)
    expect(pointCount(sequence)).toBe(counts.pointCount)
  })

  it('agrees with the delivered payload for every stroke in the document', () => {
    const layers = [
      layer('a', { operations: [stroke('s1', 12), stroke('s2', 1, 'eraser')] }),
      layer('b', { opacity: 0.5, operations: [stroke('s3', 30, 'monoline'), raster('r1')] }),
      layer('c', { visible: false, operations: [stroke('s4', 99)] }),
    ]
    const counts = documentCounts(layers)
    const sequence = buildStrokeSequence(layers)
    expect(counts.strokeCount).toBe(sequence.strokes.length)
    expect(counts.pointCount).toBe(pointCount(sequence))
    expect(counts.rasterCount).toBe(1)
  })

  it('is all zero for an empty canvas', () => {
    const layers = [layer('empty')]
    expect(documentCounts(layers)).toEqual({ strokeCount: 0, pointCount: 0, rasterCount: 0 })
    expect(buildStrokeSequence(layers).strokes).toEqual([])
  })
})

describe('buildStrokeSequence', () => {
  it('keeps the canvas it describes and rebases timestamps', () => {
    const sequence = buildStrokeSequence([layer('a', { operations: [stroke('s1', 3)] })])
    expect(sequence).toMatchObject({ version: 1, canvas_width: LOGICAL_SIZE, canvas_height: LOGICAL_SIZE })
    expect(sequence.strokes[0]?.points.map((point) => point.t)).toEqual([0, 16, 32])
  })
})
