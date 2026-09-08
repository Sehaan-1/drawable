import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SearchResult } from '@drawable/contracts'
import { matchLabel, toReferenceAsset } from './liveServices'
import { ApiError } from './apiClient'

const result = (over: Partial<SearchResult>): SearchResult => ({
  asset_id: 'ls_synthetic_ac1f55b7390698a7',
  thumbnail_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/thumbnail',
  style: 'manga_anime',
  scopes: ['eye'],
  primary_scope: 'eye',
  secondary_scopes: [],
  person_count_approximate: false,
  origin: 'native_line_art',
  trace_allowed: true,
  relevance: 0.82,
  quality: 0.7,
  asset_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/line-art',
  ...over,
})

describe('live search mapping', () => {
  it('copies trace_allowed from the API instead of hardcoding true', () => {
    const native = toReferenceAsset(result({ origin: 'native_line_art', trace_allowed: true }))
    const extracted = toReferenceAsset(
      result({ origin: 'extracted_line_art', trace_allowed: false, style: 'cartoon' }),
    )
    expect(native.traceAllowed).toBe(true)
    expect(native.native).toBe(true)
    expect(extracted.traceAllowed).toBe(false)
    expect(extracted.native).toBe(false)
  })

  it('labels match strength from relevance', () => {
    expect(matchLabel(0.81)).toBe('Strong')
    expect(matchLabel(0.6)).toBe('Close')
    expect(matchLabel(0.4)).toBe('Related')
  })
})

import { toSearchResponse } from './liveServices'
import type { SearchResponse as ApiSearchResponse } from '@drawable/contracts'
import type { SearchRequest } from '../lib/types'

const request: SearchRequest = {
  sessionId: 's', revision: 4, generation: 2, strokeCount: 3, pointCount: 6, rasterCount: 0, textHint: '', selectedStyle: null,
}
const wire = (over: Partial<ApiSearchResponse>): ApiSearchResponse => ({
  schema_version: 2,
  request_id: '3fa85f64-5717-4562-b3fc-2c963f66afa6',
  api_version: '0.1.0-test',
  revision: 4,
  canvas_width: 2048,
  canvas_height: 2048,
  stroke_status: 'present',
  counts_approximate: false,
  preprocessing_version: '1.0.0',
  mode: 'confident',
  scope_predictions: [],
  groups: [],
  timing: { preprocessing_ms: 1, embedding_ms: 0, retrieval_ms: 1, reranking_ms: 0, total_ms: 2 },
  warning: null,
  degradations: [],
  dataset_version: '2026.09.08-synthetic',
  index_version: 'abc',
  ...over,
})

describe('live search provenance round trip', () => {
  it('surfaces exact-count provenance and server versions on the view model', () => {
    const view = toSearchResponse(wire({ counts_approximate: false, stroke_status: 'present' }), request)
    expect(view.countsApproximate).toBe(false)
    expect(view.strokeStatus).toBe('present')
    expect(view.preprocessingVersion).toBe('1.0.0')
    expect(view.apiVersion).toBe('0.1.0-test')
  })

  it('flags raster-only queries as approximate', () => {
    const view = toSearchResponse(wire({ counts_approximate: true, stroke_status: 'absent' }), request)
    expect(view.countsApproximate).toBe(true)
    expect(view.strokeStatus).toBe('absent')
  })
})

import { toPinSnapshot, toPinnedReferenceAsset, liveServices } from './liveServices'
import type { PinnedAsset, PinsResponse } from '@drawable/contracts'
import { api } from './apiClient'
import { canTrace } from '../lib/trace'

const pinned = (over: Partial<PinnedAsset> = {}): PinnedAsset => ({
  asset_id: 'ls_synthetic_ac1f55b7390698a7',
  thumbnail_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/thumbnail',
  asset_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/line-art',
  style: 'manga_anime',
  scopes: ['eye'],
  primary_scope: 'eye',
  secondary_scopes: [],
  person_count: 1,
  person_count_approximate: false,
  origin: 'native_line_art',
  trace_allowed: true,
  quality: 0.7,
  pinned_at: '2026-09-08T10:00:00Z',
  ...over,
})

describe('live pin mapping', () => {
  it('keeps the server pin order, timestamp, and per-asset trace permission', () => {
    const response: PinsResponse = {
      schema_version: 1,
      gallery_kind: 'live',
      pins: [
        pinned(),
        // Extracted line art whose source explicitly permits tracing.
        pinned({ asset_id: 'ls_b', origin: 'extracted_line_art', trace_allowed: true, asset_url: '/api/v1/assets/ls_b/line-art' }),
        // Native line art the source forbids tracing.
        pinned({ asset_id: 'ls_c', origin: 'native_line_art', trace_allowed: false }),
      ],
      revoked: [{ asset_id: 'ls_gone', reasons: ['display_not_permitted'] }],
    }
    const snapshot = toPinSnapshot(response)
    expect(snapshot.pins.map((pin) => pin.id)).toEqual(['ls_synthetic_ac1f55b7390698a7', 'ls_b', 'ls_c'])
    expect(snapshot.pins[0]?.pinnedAt).toBe('2026-09-08T10:00:00Z')
    // Origin is presentation; permission decides tracing, and the two disagree here.
    expect(snapshot.pins.map(canTrace)).toEqual([true, true, false])
    expect(snapshot.pins[1]?.native).toBe(false)
    expect(snapshot.pins[2]?.native).toBe(true)
    expect(snapshot.pins[2]?.traceUrl).toBeNull()
    expect(snapshot.revoked).toEqual([{ assetId: 'ls_gone', reasons: ['display_not_permitted'] }])
  })

  it('tolerates a response without a revoked list', () => {
    expect(toPinSnapshot({ schema_version: 1, gallery_kind: 'live', pins: [], revoked: [] }).revoked).toEqual([])
  })

  it('projects a pinned asset with the same shape as a search result', () => {
    const asset = toPinnedReferenceAsset(pinned())
    expect(asset.imageUrl).toBe('/api/v1/assets/ls_synthetic_ac1f55b7390698a7/thumbnail')
    expect(asset.fullImageUrl).toBe('/api/v1/assets/ls_synthetic_ac1f55b7390698a7/line-art')
    expect(asset.style).toBe('Manga / anime')
  })
})

describe('trace resolution', () => {
  afterEach(() => vi.restoreAllMocks())

  it('asks the server and returns the trace url only when permitted', async () => {
    const spy = vi.spyOn(api, 'assetPermissions').mockResolvedValue({
      asset_id: 'ls_a', allowed_display: true, allowed_trace: true,
      permission_basis: 'public_domain', origin: 'extracted_line_art', attribution: null,
      attribution_required: false, thumbnail_url: '/t', asset_url: '/a',
      trace_url: '/api/v1/assets/ls_a/line-art',
    })
    await expect(liveServices.assets?.resolveTrace('ls_a')).resolves.toBe('/api/v1/assets/ls_a/line-art')
    expect(spy).toHaveBeenCalledWith('ls_a', undefined)
  })

  it('refuses a revoked asset even if the server still returns a url', async () => {
    vi.spyOn(api, 'assetPermissions').mockResolvedValue({
      asset_id: 'ls_a', allowed_display: true, allowed_trace: false,
      permission_basis: 'public_domain', origin: 'native_line_art', attribution: null,
      attribution_required: false, thumbnail_url: '/t', asset_url: '/a',
      trace_url: '/api/v1/assets/ls_a/line-art',
    })
    await expect(liveServices.assets?.resolveTrace('ls_a')).resolves.toBeNull()
  })

  it('fails closed when the asset is gone', async () => {
    vi.spyOn(api, 'assetPermissions').mockRejectedValue(
      new ApiError(404, 'asset_not_found', 'gone'),
    )
    await expect(liveServices.assets?.resolveTrace('ls_a')).resolves.toBeNull()
  })
})

describe('live sufficiency and degradation mapping', () => {
  const blankRaster = { kind: 'blank_raster' as const, detail: 'snapshot carries no ink' }
  const vectorAbsent = { kind: 'vector_absent' as const, detail: 'no vector stroke data' }
  const vectorSparse = { kind: 'vector_sparse' as const, detail: 'vector branch degraded' }

  it('maps a blank snapshot onto the empty state instead of "keep drawing"', () => {
    const view = toSearchResponse(
      wire({ mode: 'insufficient', degradations: [blankRaster, vectorAbsent] }),
      request,
    )
    expect(view.mode).toBe('empty')
    expect(view.interpretation).toBe('Blank canvas')
  })

  it('keeps "keep drawing" for ink that is merely too small', () => {
    const view = toSearchResponse(wire({ mode: 'insufficient', degradations: [vectorAbsent] }), request)
    expect(view.mode).toBe('insufficient')
    expect(view.interpretation).toBe('Keep drawing')
    // The absent branch is still disclosed on a degraded response.
    expect(view.degradations?.map((item) => item.kind)).toEqual(['vector_absent'])
  })

  it('never vetoes a ranked response because the vector branch is thin', () => {
    // A substantive PNG/SVG import: the server ranked it, so the client shows
    // those results and surfaces the missing geometry as a structural note —
    // vector availability never overrides the gallery's sufficiency verdict.
    const view = toSearchResponse(
      wire({
        mode: 'confident',
        stroke_status: 'absent',
        counts_approximate: true,
        groups: [
          { id: 'best', kind: 'best_match', title: 'Best match', results: [result({})] },
        ],
        degradations: [vectorAbsent],
        warning: vectorAbsent.detail,
      }),
      request,
    )
    expect(view.mode).toBe('confident')
    expect(view.groups).toHaveLength(1)
    expect(view.groups[0]?.results[0]?.id).toBe('ls_synthetic_ac1f55b7390698a7')
    expect(view.strokeStatus).toBe('absent')
    expect(view.degradations?.map((item) => item.kind)).toEqual(['vector_absent'])
    expect(view.warning).toBe(vectorAbsent.detail)
  })

  it('reports a blank import as empty even when it also lost the vector branch', () => {
    const view = toSearchResponse(
      wire({ mode: 'insufficient', degradations: [blankRaster, vectorAbsent], groups: [] }),
      request,
    )
    expect(view.mode).toBe('empty')
    expect(view.degradations?.map((item) => item.kind)).toEqual(['blank_raster', 'vector_absent'])
  })

  it('carries sparse-vector disclosure through untouched', () => {
    const view = toSearchResponse(wire({ mode: 'provisional', degradations: [vectorSparse] }), request)
    expect(view.mode).toBe('provisional')
    expect(view.degradations?.[0]?.detail).toBe('vector branch degraded')
  })

  it('tolerates a server that predates the degradations field', () => {
    const view = toSearchResponse({ ...wire({ mode: 'confident' }), degradations: undefined }, request)
    expect(view.degradations).toEqual([])
    expect(view.mode).toBe('confident')
  })
})
