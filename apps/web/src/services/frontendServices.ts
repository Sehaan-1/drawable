import { fixtureAssets, fixtureHealth, fixtureSearch } from './fixtures'
import type { HealthResult, SearchRequest, SearchResponse } from '../lib/types'
import type { components } from '@drawable/contracts'

type ApiSearchResponse = components['schemas']['SearchResponse']
const searchSessionId = crypto.randomUUID()
const documentIds = new Map<string, string>()

export function isRealSearchMode() {
  const mode = new URLSearchParams(window.location.search).get('mode')
  return mode === 'real' || (mode !== 'fixture' && import.meta.env.VITE_DRAWABLE_SEARCH_MODE === 'real')
}

async function realSearch(request: SearchRequest, signal: AbortSignal): Promise<SearchResponse> {
  if (!request.image || !request.documentId) throw new Error('The drawing snapshot is unavailable. Try again.')
  let documentId = documentIds.get(request.documentId)
  if (!documentId) {
    documentId = crypto.randomUUID()
    documentIds.set(request.documentId, documentId)
  }
  const fields = new FormData()
  const values = {
    session_id: searchSessionId, document_id: documentId,
    canvas_revision: String(request.revision), query_generation: String(request.generation),
    canvas_width: '2048', canvas_height: '2048', stroke_count: String(request.strokeCount),
    point_count: '0', counts_approximate: 'true', stroke_status: 'disabled',
    preprocessing_version: 'drawable-query-v1', text_hint: '',
  }
  for (const [key, value] of Object.entries(values)) fields.set(key, value)
  fields.set('image', request.image, 'query.png')
  const response = await fetch('/api/v1/search', { method: 'POST', body: fields, signal, credentials: 'same-origin', cache: 'no-store' })
  const value = await response.json() as ApiSearchResponse & { error?: { code: string; message: string } }
  if (!response.ok) {
    if (response.status === 409 && ['superseded_search', 'stale_generation'].includes(value.error?.code ?? '')) throw new DOMException('Search superseded', 'AbortError')
    throw new Error(value.error?.message ?? `Local reference search failed (${response.status}).`)
  }
  if (value.session_id !== searchSessionId || value.document_id !== documentId || value.canvas_revision !== request.revision || value.query_generation !== request.generation) throw new DOMException('Search superseded', 'AbortError')
  return {
    revision: value.canvas_revision, generation: value.query_generation, mode: value.mode,
    interpretation: value.interpretation, buildId: value.build_id ?? undefined,
    gallerySize: value.gallery_size,
    scopeCoverage: value.scope_coverage,
    styleCoverage: value.style_coverage,
    groups: (value.groups ?? []).map((group) => {
      return { id: group.id, title: group.title, tentative: group.tentative, results: group.results.map((asset) => ({ ...asset, imageUrl: asset.image_url, traceAllowed: asset.trace_allowed })) }
    }),
  }
}

export interface SearchClient {
  search(request: SearchRequest, signal: AbortSignal): Promise<SearchResponse>
}

export interface AssetClient {
  resolveTrace(assetId: string, signal?: AbortSignal): Promise<string | null>
}

export interface FrontendServices {
  health: { get(): Promise<HealthResult> }
  search: SearchClient
  assets: AssetClient
  preferences: Record<string, never>
  pins: Record<string, never>
  events: Record<string, never>
  curation: Record<string, never>
  benchmarks: Record<string, never>
}

export const fixtureServices: FrontendServices = {
  health: { get: async () => fixtureHealth },
  search: { search: fixtureSearch },
  assets: { resolveTrace: async (assetId) => fixtureAssets.find((asset) => asset.id === assetId)?.imageUrl ?? null },
  preferences: {},
  pins: {},
  events: {},
  curation: {},
  benchmarks: {},
}

export const realServices: FrontendServices = {
  ...fixtureServices,
  health: { get: async () => {
    const response = await fetch('/api/v1/health', { cache: 'no-store' })
    const value = await response.json() as components['schemas']['HealthResponse']
    return { mode: value.mode, ready: value.ready, message: value.ready ? `${value.gallery_size} reviewed references · local CPU` : 'Local reference search needs setup.' }
  } },
  search: { search: realSearch },
  assets: { resolveTrace: async () => null },
}

export function getFrontendServices() {
  return isRealSearchMode() ? realServices : fixtureServices
}
