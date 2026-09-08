import { describe, expect, it } from 'vitest'
import type { SearchResult } from '@drawable/contracts'
import { matchLabel, toReferenceAsset } from './liveServices'

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

describe('live search provenance round trip', () => {
  const request: SearchRequest = {
    sessionId: 's', revision: 4, generation: 2, strokeCount: 3, pointCount: 6, textHint: '', selectedStyle: null,
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
