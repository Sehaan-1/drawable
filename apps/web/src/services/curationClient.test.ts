import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from './apiClient'
import {
  adjudicateSfw,
  createCrop,
  exportSnapshot,
  fetchCandidate,
  fetchCurationProgress,
  fetchNextCandidate,
  fetchQuarantine,
  processDerivative,
  revealQuarantined,
  skipCandidate,
  writeLabel,
} from './curationClient'

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
  vi.restoreAllMocks()
})

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('fetchNextCandidate', () => {
  it('returns the candidate on a 200', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockJsonResponse({ asset_id: 'a', primary_style: 'manga_anime' }),
    ) as unknown as typeof fetch
    const result = await fetchNextCandidate({ style: 'manga_anime' })
    expect(result?.asset_id).toBe('a')
  })

  it('returns null on a 404 with code queue_empty', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockJsonResponse({ error: { code: 'queue_empty', message: 'no candidates' } }, 404),
    ) as unknown as typeof fetch
    expect(await fetchNextCandidate({})).toBeNull()
  })

  it('returns null on a 404 with code gallery_unavailable', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockJsonResponse({ error: { code: 'gallery_unavailable', message: 'no gallery' } }, 404),
    ) as unknown as typeof fetch
    expect(await fetchNextCandidate({})).toBeNull()
  })

  it('throws on a 404 with a different error code', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockJsonResponse({ error: { code: 'asset_not_found', message: 'nope' } }, 404),
    ) as unknown as typeof fetch
    await expect(fetchNextCandidate({})).rejects.toBeInstanceOf(ApiError)
  })

  it('appends the style and scope to the query string', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockJsonResponse({ asset_id: 'a' })) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    await fetchNextCandidate({ style: 'cartoon', scope: 'eye' })
    const url = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0] as string
    expect(url).toContain('style=cartoon')
    expect(url).toContain('scope=eye')
  })
})

describe('fetchCurationProgress', () => {
  it('hits /curation/progress', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ reviewed: 1, accepted: 1, rejected: 0, remaining: 1999, target: 2000 }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    const result = await fetchCurationProgress()
    expect(result.reviewed).toBe(1)
    expect((fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0]).toBe(
      '/api/v1/curation/progress',
    )
  })
})

describe('writeLabel', () => {
  it('POSTs JSON to /curation/labels', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ id: 1, asset_id: 'a', decision: 'keep', review_state: 'accepted' }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    await writeLabel({
      asset_id: 'a',
      expected_review_state: 'unreviewed',
      expected_label_version: 0,
      decision: 'keep',
      primary_scope: 'eye',
      secondary_scopes: [],
      blockers: [],
      quality: 3,
      sfw_safe: true,
    })
    const [url, init] = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0] ?? []
    expect(url).toBe('/api/v1/curation/labels')
    expect((init as RequestInit).method).toBe('POST')
    expect(JSON.parse((init as RequestInit).body as string)).toMatchObject({ decision: 'keep' })
  })
})

describe('fetchNextCandidate with a session', () => {
  it('appends session_id to the query string', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockJsonResponse({ asset_id: 'a' })) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    await fetchNextCandidate({ sessionId: 'sess-1' })
    const url = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0] as string
    expect(url).toContain('session_id=sess-1')
  })
})

describe('fetchCandidate', () => {
  it('fetches one candidate by id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ asset_id: 'a', review_state: 'accepted', label_version: 2 }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    const result = await fetchCandidate('a')
    expect(result.asset_id).toBe('a')
    expect(
      (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0],
    ).toBe('/api/v1/curation/candidates/a')
  })
})

describe('skipCandidate', () => {
  it('POSTs the session skip', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ session_id: 's', cursor_asset_id: 'a', excluded_asset_id: 'a', remaining: 3 }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    const result = await skipCandidate({ session_id: 's', asset_id: 'a' })
    expect(result.remaining).toBe(3)
    const [url, init] = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0] ?? []
    expect(url).toBe('/api/v1/curation/queue/skip')
    expect(JSON.parse((init as RequestInit).body as string)).toMatchObject({
      session_id: 's',
      asset_id: 'a',
    })
  })
})

describe('quarantine + SFW adjudication', () => {
  it('fetches the metadata-only backlog', async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockJsonResponse([])) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    await fetchQuarantine()
    expect((fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0]).toBe(
      '/api/v1/curation/quarantine',
    )
  })

  it('reveals a held record', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ asset_id: 'a', revealed: true }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    const result = await revealQuarantined('a', { reviewer: 'me' })
    expect(result.revealed).toBe(true)
    expect((fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0]).toBe(
      '/api/v1/curation/quarantine/a/reveal',
    )
  })

  it('posts an adjudication with the expected label version', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ id: 1, asset_id: 'a', safe: true }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    await adjudicateSfw('a', { safe: true, expected_label_version: 1 })
    const [url, init] = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0] ?? []
    expect(url).toBe('/api/v1/curation/sfw/a/adjudication')
    expect(JSON.parse((init as RequestInit).body as string)).toMatchObject({
      safe: true,
      expected_label_version: 1,
    })
  })
})

describe('crop derivatives', () => {
  it('creates a crop derivative from the parent asset', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ asset_id: 'child', parent_asset_id: 'a', processing_state: 'pending' }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    const result = await createCrop('a', {
      crop: { x: 0, y: 0, width: 64, height: 64 },
      expected_label_version: 2,
    })
    expect(result.processing_state).toBe('pending')
    const [url, init] = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0] ?? []
    expect(url).toBe('/api/v1/curation/assets/a/crops')
    expect(JSON.parse((init as RequestInit).body as string)).toMatchObject({
      expected_label_version: 2,
    })
  })

  it('processes a derivative', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ asset_id: 'child', processing_state: 'complete' }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    await processDerivative('child')
    expect((fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls[0]?.[0]).toBe(
      '/api/v1/curation/assets/child/process',
    )
  })
})

describe('exportSnapshot', () => {
  it('POSTs to /curation/snapshots', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      mockJsonResponse({ snapshot_id: 'curation_x', path: 'snapshots/curation_x.json', label_count: 0, style_breakdown: {}, created_at: 'x' }),
    ) as unknown as typeof fetch
    globalThis.fetch = fetchMock
    const result = await exportSnapshot()
    expect(result.snapshot_id).toBe('curation_x')
  })
})
