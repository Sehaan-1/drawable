import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fixtureSearch } from './fixtures'
import { measureInk, vectorBranchStatus, type RasterInk } from '../lib/rasterInk'
import type { SearchRequest } from '../lib/types'

/** Ink measured on a 512² snapshot: a real drawing, and a canvas with nothing on it. */
const SUBSTANTIVE: RasterInk = { inkPixels: 4096, coverage: 4096 / (512 * 512), bboxDiagonalRatio: 0.62 }
const DOT: RasterInk = { inkPixels: 12, coverage: 12 / (512 * 512), bboxDiagonalRatio: 0.004 }
const BLANK: RasterInk = { inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 }

const req = (over: Partial<SearchRequest>): SearchRequest => ({ sessionId: 's', revision: 1, generation: 1, strokeCount: 0, pointCount: 0, rasterCount: 0, textHint: '', selectedStyle: null, ...over })

describe('fixture reference search', () => {
  it('progresses from empty to provisional to confident', async () => {
    const empty = await fixtureSearch(req({ revision: 0, generation: 1, strokeCount: 0 }), new AbortController().signal)
    const provisional = await fixtureSearch(req({ revision: 1, generation: 2, strokeCount: 1 }), new AbortController().signal)
    const confident = await fixtureSearch(req({ revision: 3, generation: 3, strokeCount: 3 }), new AbortController().signal)
    expect(empty.mode).toBe('empty')
    expect(provisional.mode).toBe('provisional')
    expect(confident.mode).toBe('confident')
    expect(confident.groups[0]?.title).toBe('Best match')
    const assets = confident.groups.flatMap((group) => group.results)
    expect(assets.some((asset) => asset.traceAllowed)).toBe(true)
    expect(assets.some((asset) => !asset.traceAllowed)).toBe(true)
    // Permission is never inferred from native/extracted origin: the fixture
    // gallery contains both mismatched combinations, and a forbidden asset
    // never carries a trace source.
    expect(assets.some((asset) => asset.native && !asset.traceAllowed)).toBe(true)
    expect(assets.some((asset) => !asset.native && asset.traceAllowed)).toBe(true)
    expect(assets.every((asset) => (asset.traceUrl === null) === !asset.traceAllowed)).toBe(true)
  })

  it('honors request cancellation', async () => {
    const controller = new AbortController()
    const result = fixtureSearch(req({ revision: 1, generation: 2, strokeCount: 1, textHint: 'slow' }), controller.signal)
    controller.abort()
    await expect(result).rejects.toMatchObject({ name: 'AbortError' })
  })
})

describe('fixture sufficiency is measured on ink', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  /** Drives the fixture's debounce deterministically instead of sleeping. */
  async function search(request: SearchRequest) {
    const pending = fixtureSearch(request, new AbortController().signal)
    await vi.advanceTimersByTimeAsync(500)
    return pending
  }

  it('spends nothing on a blank canvas and says so structurally', async () => {
    const response = await search(req({ strokeCount: 0, pointCount: 0, rasterCount: 1, ink: BLANK }))
    expect(response.mode).toBe('empty')
    expect(response.interpretation).toBe('Blank canvas')
    expect(response.degradations?.map((item) => item.kind)).toEqual(['blank_raster', 'vector_absent'])
    expect(response.warning).toContain('no ink')
  })

  it('treats a fully transparent import as blank, not as a small drawing', async () => {
    // A transparent PNG has no alpha-carrying pixels at all, so measuring it
    // yields the same verdict as an untouched canvas — which is the point: the
    // old count-based rule called it "searchable" because an operation existed.
    const transparent = measureInk(new Uint8ClampedArray(64), 4, 4)
    expect(transparent).toEqual(BLANK)
    const response = await search(req({ rasterCount: 1, ink: transparent }))
    expect(response.mode).toBe('empty')
  })

  it('searches a substantive raster import that carries no vector geometry', async () => {
    const response = await search(
      req({ strokeCount: 0, pointCount: 0, rasterCount: 1, ink: SUBSTANTIVE }),
    )
    // Real ink means real results, even with an empty stroke branch; the branch
    // is disclosed as absent rather than used to veto the query.
    expect(response.mode).not.toBe('empty')
    expect(response.groups.length).toBeGreaterThan(0)
    expect(response.degradations?.map((item) => item.kind)).toEqual(['vector_absent'])
    expect(response.strokeStatus).toBe('absent')
    expect(response.countsApproximate).toBe(true)
  })

  it('does not demote an adequate raster for thin vector counts', async () => {
    const response = await search(
      req({ strokeCount: 3, pointCount: 12, rasterCount: 0, ink: SUBSTANTIVE }),
    )
    expect(response.mode).toBe('confident')
    expect(response.degradations?.map((item) => item.kind)).toEqual(['vector_sparse'])
  })

  it('calls a smudge insufficient while keeping the vector branch separate', async () => {
    const response = await search(req({ strokeCount: 9, pointCount: 900, ink: DOT }))
    expect(response.mode).toBe('insufficient')
    expect(response.interpretation).toBe('Too little ink to read a subject from')
    // 900 points is a healthy stroke branch; the *drawing* is what is too small.
    expect(vectorBranchStatus(9, 900)).toBe('usable')
    expect(response.degradations ?? []).toEqual([])
  })

  it('falls back to counts when pixels could not be measured at all', async () => {
    // jsdom and privacy-restricted canvases yield no image data. Guessing
    // "blank" there would silence search for a real drawing, so an unmeasurable
    // snapshot only goes empty when the vector branch is empty too.
    const unmeasurable = await search(req({ strokeCount: 5, pointCount: 400, ink: null }))
    expect(unmeasurable.mode).toBe('confident')
    expect(unmeasurable.degradations ?? []).toEqual([])
    const unmeasurableBlank = await search(req({ strokeCount: 0, pointCount: 0, ink: null }))
    expect(unmeasurableBlank.mode).toBe('empty')
  })

  it('keeps an explicit no-results hint distinct from a too-small drawing', async () => {
    const response = await search(req({ strokeCount: 5, pointCount: 400, ink: SUBSTANTIVE, textHint: 'empty' }))
    expect(response.mode).toBe('insufficient')
    expect(response.interpretation).toBe('No relevant candidates')
  })
})
