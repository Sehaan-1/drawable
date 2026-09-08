import { describe, expect, it } from 'vitest'
import {
  INK_THRESHOLD,
  inputDegradations,
  measureInk,
  rasterSufficiency,
  vectorBranchStatus,
  type RasterInk,
} from './rasterInk'

/**
 * The client's copy of the API's sufficiency rule, measured on the snapshot
 * rather than inferred from vector counts. These tests are deliberately about
 * the two cases the counts get wrong: a substantive import with no vectors, and
 * a transparent import with none either.
 */

const SIDE = 100

type Pixel = [number, number, number, number]

function canvas(paint: (x: number, y: number) => Pixel | null): Uint8ClampedArray {
  const data = new Uint8ClampedArray(SIDE * SIDE * 4)
  for (let y = 0; y < SIDE; y += 1) {
    for (let x = 0; x < SIDE; x += 1) {
      const pixel = paint(x, y) ?? ([255, 255, 255, 255] satisfies Pixel)
      const offset = (y * SIDE + x) * 4
      data[offset] = pixel[0]
      data[offset + 1] = pixel[1]
      data[offset + 2] = pixel[2]
      data[offset + 3] = pixel[3]
    }
  }
  return data
}

const WHITE: Pixel = [255, 255, 255, 255]
const INK: Pixel = [17, 18, 20, 255]

function ink(over: Partial<RasterInk>): RasterInk {
  return { inkPixels: 4096, coverage: 0.4, bboxDiagonalRatio: 0.6, ...over }
}

describe('measureInk', () => {
  it('sees nothing in a fully transparent canvas, even where it is black', () => {
    // Transparency is flattened onto white before the API measures, so a black
    // but fully transparent pixel is *not* ink. This is the blank-import case.
    const transparent = canvas(() => [0, 0, 0, 0])
    expect(measureInk(transparent, SIDE, SIDE)).toEqual({ inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 })
  })

  it('counts a white canvas as blank too', () => {
    expect(measureInk(canvas(() => WHITE), SIDE, SIDE).inkPixels).toBe(0)
  })

  it('measures spread from the ink bounding box, not from the mark count', () => {
    const dot = measureInk(canvas((x, y) => (x === 49 && y === 49 ? INK : WHITE)), SIDE, SIDE)
    expect(dot.inkPixels).toBe(1)
    // A single pixel has no extent: a dot is not a drawing, and it never
    // reaches the sufficiency floor no matter how many strokes claim to exist.
    expect(dot.bboxDiagonalRatio).toBeCloseTo(Math.hypot(1, 1) / Math.hypot(SIDE, SIDE))
    expect(rasterSufficiency(dot).insufficient).toBe(true)
    expect(rasterSufficiency(dot).blank).toBe(false)

    const stroke = measureInk(canvas((x, y) => (y === 10 && x >= 5 && x < 95 ? INK : WHITE)), SIDE, SIDE)
    expect(stroke.inkPixels).toBe(90)
    expect(rasterSufficiency(stroke).insufficient).toBe(false)
    expect(rasterSufficiency(stroke).blank).toBe(false)
  })

  it('treats a gray value at the threshold as background', () => {
    const atThreshold = measureInk(canvas(() => [INK_THRESHOLD, INK_THRESHOLD, INK_THRESHOLD, 255]), SIDE, SIDE)
    expect(atThreshold.inkPixels).toBe(0)
    const justBelow = measureInk(canvas(() => [INK_THRESHOLD - 1, INK_THRESHOLD - 1, INK_THRESHOLD - 1, 255]), SIDE, SIDE)
    expect(justBelow.inkPixels).toBe(SIDE * SIDE)
  })
})

describe('rasterSufficiency', () => {
  it('separates "nothing was drawn" from "too little was drawn"', () => {
    expect(rasterSufficiency(ink({ inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 }))).toMatchObject({
      blank: true,
      insufficient: true,
      unknown: false,
    })
    expect(rasterSufficiency(ink({ inkPixels: 4, coverage: 0.0004, bboxDiagonalRatio: 0.001 }))).toMatchObject({
      blank: false,
      insufficient: true,
    })
    expect(rasterSufficiency(ink({}))).toMatchObject({ blank: false, insufficient: false })
  })

  it('reports an unmeasurable snapshot as unknown, never as blank', () => {
    // Guessing "blank" here would silence search for a drawing the client could
    // not measure; the caller has to fall back to another signal instead.
    expect(rasterSufficiency(null)).toEqual({ blank: false, insufficient: false, unknown: true })
  })
})

describe('vectorBranchStatus', () => {
  it('is absent only when there is no vector input at all', () => {
    expect(vectorBranchStatus(0, 0)).toBe('absent')
    // A substantive PNG/SVG import: no strokes, no points, real ink.
    expect(vectorBranchStatus(0, 0)).toBe('absent')
    expect(vectorBranchStatus(1, 0)).toBe('sparse')
  })

  it('calls a small nonzero count sparse, not absent', () => {
    expect(vectorBranchStatus(1, 19)).toBe('sparse')
    expect(vectorBranchStatus(14, 20)).toBe('usable')
    expect(vectorBranchStatus(14, 900)).toBe('usable')
  })

  it('prefers the delivered payload over the reported estimate', () => {
    // The client claims 900 points but is sending 6: the branch is judged on
    // what actually arrives, exactly as the API judges it.
    expect(vectorBranchStatus(3, 900, 6)).toBe('sparse')
    expect(vectorBranchStatus(3, 0, 24)).toBe('usable')
  })
})

describe('inputDegradations', () => {
  it('labels a blank snapshot structurally', () => {
    const items = inputDegradations(ink({ inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 }), 0, 0)
    expect(items.map((item) => item.kind)).toEqual(['blank_raster', 'vector_absent'])
    expect(items[0]?.detail).toContain('no ink')
  })

  it('does not call a substantive import blank', () => {
    const items = inputDegradations(ink({}), 0, 0)
    expect(items.map((item) => item.kind)).toEqual(['vector_absent'])
  })

  it('discloses a thin vector payload as a degraded branch, with the count it saw', () => {
    const items = inputDegradations(ink({}), 1, 19)
    expect(items).toHaveLength(1)
    expect(items[0]?.kind).toBe('vector_sparse')
    expect(items[0]?.detail).toContain('19')
    expect(items[0]?.detail).toContain('client-reported count')

    const measured = inputDegradations(ink({}), 3, 900, 6)
    expect(measured[0]?.detail).toContain('6')
    expect(measured[0]?.detail).not.toContain('client-reported count')
  })

  it('is empty when both halves of the query are complete', () => {
    expect(inputDegradations(ink({}), 14, 900, 900)).toEqual([])
  })
})
