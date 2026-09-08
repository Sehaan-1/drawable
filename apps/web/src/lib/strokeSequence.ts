import type { StrokeSequence } from '@drawable/contracts'
import { LOGICAL_SIZE, type DrawingLayer, type RasterOperation, type StrokeOperation } from './types'

/**
 * Flatten visible layers into the wire-format stroke sequence.
 *
 * Raw points are sent untouched (the server resamples/simplifies); timestamps
 * are rebased to the first sample so the payload does not leak wall-clock
 * time. Erase operations are included so the server can reproduce the raster.
 * Raster artwork (imported images) is excluded from stroke sequences.
 *
 * The counts on the wire describe *this* payload and nothing else:
 * `stroke_count` is validated against the delivered array element-wise by the
 * API (a mismatch is a 422, not a warning), so counting any operation the
 * sequence omits — an import, or ink on a hidden or fully transparent layer —
 * would break the request. `visibleStrokeOperations` is therefore the single
 * source of truth for both the payload and the counts.
 */
export function visibleStrokeOperations(layers: DrawingLayer[]): StrokeOperation[] {
  return layers
    .filter((layer) => layer.visible && layer.opacity > 0)
    .flatMap((layer) => layer.operations)
    .filter((operation): operation is StrokeOperation => operation.kind === 'stroke')
}

function visibleRasterOperations(layers: DrawingLayer[]): RasterOperation[] {
  return layers
    .filter((layer) => layer.visible && layer.opacity > 0)
    .flatMap((layer) => layer.operations)
    .filter((operation): operation is RasterOperation => operation.kind === 'raster')
}

export function buildStrokeSequence(layers: DrawingLayer[]): StrokeSequence {
  const operations = visibleStrokeOperations(layers)
  const origin = operations[0]?.points[0]?.time ?? 0
  return {
    version: 1,
    canvas_width: LOGICAL_SIZE,
    canvas_height: LOGICAL_SIZE,
    strokes: operations.map((operation) => ({
      tool: operation.tool,
      pointer: operation.points[0]?.pointerType ?? (operation.simulatePressure ? 'mouse' : 'pen'),
      points: operation.points.map((point) => ({
        x: point.x,
        y: point.y,
        p: Math.min(1, Math.max(0, point.pressure)),
        t: Math.max(0, point.time - origin),
      })),
    })),
  }
}

export interface DocumentCounts {
  /** Strokes in the delivered vector payload (0 for a raster-only import). */
  strokeCount: number
  /** Sampled points across those strokes. */
  pointCount: number
  /**
   * Imported raster operations. These carry no vector geometry, so they are
   * counted separately: they say the drawing has *content* (measured as ink on
   * the snapshot) without inflating the stroke counts the payload is checked
   * against.
   */
  rasterCount: number
}

/** Counts for one document, derived from the exact operations the payload uses. */
export function documentCounts(layers: DrawingLayer[]): DocumentCounts {
  const strokes = visibleStrokeOperations(layers)
  return {
    strokeCount: strokes.length,
    pointCount: strokes.reduce((total, operation) => total + operation.points.length, 0),
    rasterCount: visibleRasterOperations(layers).length,
  }
}
