import { afterEach, describe, expect, it, vi } from 'vitest'
import { realServices } from './frontendServices'

afterEach(() => { vi.unstubAllGlobals() })

describe('local search adapter', () => {
  it('sends the actual snapshot and adapts reviewed references without enabling tracing', async () => {
    const image = new Blob(['test snapshot'], { type: 'image/png' })
    const fetchMock = vi.fn(async (_path: string, init: RequestInit) => {
      const form = init.body as FormData
      expect(form.get('canvas_width')).toBe('2048')
      expect(form.get('stroke_status')).toBe('disabled')
      expect(form.get('text_hint')).toBe('')
      expect((form.get('image') as File).size).toBe(image.size)
      return new Response(JSON.stringify({
        session_id: form.get('session_id'), document_id: form.get('document_id'),
        canvas_revision: 4, query_generation: 6, build_id: 'prototype-test',
        mode: 'provisional', interpretation: 'Tentative reviewed references',
        groups: [{ id: 'face_head', title: 'Reviewed heads', tentative: true, results: [{
          id: 'real-reference', title: 'Head sketch', image_url: '/api/v1/assets/real-reference/preview',
          style: 'Cartoon', scope: 'face_head', source: 'Recorded artist', native: true,
          match: 'Tentative', trace_allowed: false,
        }] }],
      }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    const result = await realServices.search.search({ documentId: 'local-document', image, revision: 4, generation: 6, strokeCount: 2, textHint: 'Ignored during this image-only experiment' }, new AbortController().signal)
    expect(result.mode).toBe('provisional')
    expect(result.groups[0]?.results[0]).toMatchObject({ id: 'real-reference', imageUrl: '/api/v1/assets/real-reference/preview', traceAllowed: false, match: 'Tentative' })
  })

  it('discards responses belonging to another drawing or generation', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ session_id: crypto.randomUUID(), document_id: crypto.randomUUID(), canvas_revision: 1, query_generation: 1 }), { status: 200 })))
    await expect(realServices.search.search({ documentId: 'another-document', image: new Blob(['snapshot']), revision: 1, generation: 1, strokeCount: 1, textHint: '' }, new AbortController().signal)).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('reports unavailable real search without substituting fixture images', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { code: 'search_not_configured', message: 'Prepare local search first.' } }), { status: 503 })))
    await expect(realServices.search.search({ documentId: 'local-document', image: new Blob(['snapshot']), revision: 1, generation: 2, strokeCount: 1, textHint: '' }, new AbortController().signal)).rejects.toThrow('Prepare local search first.')
  })
})
