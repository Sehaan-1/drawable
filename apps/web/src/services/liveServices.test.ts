import { describe, expect, it } from 'vitest'
import type { SearchResult } from '@drawable/contracts'
import { matchLabel, toReferenceAsset } from './liveServices'

const result = (over: Partial<SearchResult>): SearchResult => ({
  asset_id: 'ls_synthetic_ac1f55b7390698a7',
  thumbnail_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/thumbnail',
  style: 'manga_anime',
  scopes: ['eye'],
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
