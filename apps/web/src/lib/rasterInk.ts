/**
 * Raster ink measurement — the client half of the API's sufficiency rule.
 *
 * Whether a drawing is worth searching at all is decided by the *ink in the
 * snapshot*, never by how many vector points happen to be attached to it. A
 * vector-less import (a PNG or an SVG flattened into one raster operation) is a
 * real drawing; a canvas whose only strokes were erased, or an import whose
 * pixels are all transparent, is blank no matter how many operations it holds.
 *
 * `measureInk` applies the same rule the API applies to the uploaded PNG
 * (`services/api/linescout_api/preprocessing.py`: grayscale below 200 counts as
 * ink, insufficiency is ink spread against the canvas diagonal), so both sides
 * agree on what "nothing to search" means before a request is even made.
 */

/** Grayscale values below this count as ink (mirrors the API's INK_THRESHOLD). */
export const INK_THRESHOLD = 200

/** Ink bounding-box diagonal as a fraction of the snapshot diagonal. */
export const MIN_INK_DIAGONAL_RATIO = 0.02

/** Minimum sampled points before the stroke branch has anything to rank. */
export const MIN_POINTS_FOR_VECTOR_BRANCH = 20

export interface RasterInk {
  /** Pixels dark enough to be ink. */
  inkPixels: number
  /** `inkPixels / (width * height)`. */
  coverage: number
  /**
   * Ink bounding-box diagonal over the snapshot diagonal, in `[0, 1]`; `0`
   * when there is no ink at all.
   */
  bboxDiagonalRatio: number
}

export interface RasterSufficiency {
  /** Nothing was drawn, or everything drawn was erased / fully transparent. */
  blank: boolean
  /** Too little ink to read a drawing from — blank or a sub-threshold smudge. */
  insufficient: boolean
  /**
   * True when the measurement was unavailable (no readable 2D surface), so the
   * caller must fall back to another signal instead of guessing "blank".
   */
  unknown: boolean
}

/** Vector-branch availability, separate from sufficiency by design. */
export type VectorBranchStatus = 'usable' | 'sparse' | 'absent'

export function measureInk(data: Uint8ClampedArray, width: number, height: number): RasterInk {
  if (!width || !height) return { inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 }
  let inkPixels = 0
  let left = width
  let top = height
  let right = 0
  let bottom = 0
  for (let y = 0; y < height; y += 1) {
    const row = y * width
    for (let x = 0; x < width; x += 1) {
      const offset = (row + x) * 4
      // Transparency is flattened onto white by the API before it measures,
      // so a fully transparent pixel is background here too — not ink.
      if ((data[offset + 3] ?? 0) === 0) continue
      const gray = 0.2126 * (data[offset] ?? 0) + 0.7152 * (data[offset + 1] ?? 0) + 0.0722 * (data[offset + 2] ?? 0)
      if (gray >= INK_THRESHOLD) continue
      inkPixels += 1
      if (x < left) left = x
      if (x >= right) right = x + 1
      if (y < top) top = y
      if (y >= bottom) bottom = y + 1
    }
  }
  if (inkPixels === 0) return { inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 }
  return {
    inkPixels,
    coverage: inkPixels / (width * height),
    bboxDiagonalRatio: Math.min(1, Math.hypot(right - left, bottom - top) / Math.hypot(width, height)),
  }
}

/**
 * Sufficiency of a snapshot from its ink measurements.
 *
 * `null` (unmeasurable) is reported as `unknown`, never as `blank`: a client
 * that could not read pixels must not silence search for a drawing that has
 * ink, and a client that can must not search a canvas that has none.
 */
export function rasterSufficiency(ink: RasterInk | null | undefined): RasterSufficiency {
  if (!ink) return { blank: false, insufficient: false, unknown: true }
  const blank = ink.inkPixels <= 0 || ink.coverage <= 0
  return {
    blank,
    insufficient: blank || ink.bboxDiagonalRatio < MIN_INK_DIAGONAL_RATIO,
    unknown: false,
  }
}

/**
 * Whether the stroke (vector) branch has data to rank.
 *
 * `deliveredPointCount` is the exact total of a strokes payload that is really
 * being sent, which outranks the client's estimate; either way the answer only
 * describes the vector branch. A `sparse` or `absent` branch degrades results,
 * it does not make the drawing unsearchable.
 */
export function vectorBranchStatus(
  strokeCount: number,
  pointCount: number,
  deliveredPointCount?: number,
): VectorBranchStatus {
  const total = deliveredPointCount ?? pointCount
  if (strokeCount <= 0 && total <= 0) return 'absent'
  if (strokeCount <= 0 || total < MIN_POINTS_FOR_VECTOR_BRANCH) return 'sparse'
  return 'usable'
}

/**
 * The query's *input* degradations, in the same shape the API reports them.
 *
 * Kept structural (kind + detail) rather than derived in the UI: "the import
 * was blank" and "the stroke branch had too little to rank" are different
 * facts with different fixes, and a client that has to recompute them from
 * counts gets them wrong in exactly the raster-import case.
 */
export function inputDegradations(
  ink: RasterInk | null | undefined,
  strokeCount: number,
  pointCount: number,
  deliveredPointCount?: number,
): Array<{ kind: 'blank_raster' | 'vector_absent' | 'vector_sparse'; detail: string }> {
  const items: Array<{ kind: 'blank_raster' | 'vector_absent' | 'vector_sparse'; detail: string }> = []
  if (rasterSufficiency(ink).blank) {
    items.push({
      kind: 'blank_raster',
      detail: 'snapshot carries no ink: nothing was drawn, or the imported image is blank',
    })
  }
  const vector = vectorBranchStatus(strokeCount, pointCount, deliveredPointCount)
  if (vector === 'absent') {
    items.push({
      kind: 'vector_absent',
      detail: 'no vector stroke data in this query; the stroke branch contributed nothing',
    })
  }
  if (vector === 'sparse') {
    items.push({
      kind: 'vector_sparse',
      detail:
        `vector branch degraded: ${deliveredPointCount ?? pointCount} sampled point(s) below the ` +
        `${MIN_POINTS_FOR_VECTOR_BRANCH} this branch needs${deliveredPointCount === undefined ? ' (client-reported count)' : ''}`,
    })
  }
  return items
}
