/**
 * Live service implementation backed by the FastAPI worker.
 *
 * Adapts the wire contract (`@drawable/contracts`) to the UI's view models in
 * `lib/types.ts` so the reference dock does not care whether results come from
 * the fixture ranker or a real model.
 */

import {
  SCOPE_TITLES,
  STYLE_TITLES,
  type HealthResponse,
  type PinnedAsset,
  type PinsResponse,
  type PrimaryStyle,
  type SearchGroup,
  type SearchResponse as ApiSearchResponse,
  type SearchResult,
  type StyleSelection,
} from '@drawable/contracts'
import type { HealthResult, ReferenceAsset, ReferenceGroup, SearchRequest, SearchResponse } from '../lib/types'
import { LOGICAL_SIZE } from '../lib/types'
import { ApiError, api } from './apiClient'
import type { FrontendServices, PinSnapshot } from './frontendServices'
import { takeLegacyLivePinIds } from './pinStorage'

export const UI_STYLE_TO_API: Record<string, PrimaryStyle> = {
  'Manga / anime': 'manga_anime',
  'Western ink': 'western_ink',
  Realistic: 'realistic_academic',
  Cartoon: 'cartoon',
  Gesture: 'gesture_sketch',
}

const API_STYLE_TO_UI: Record<PrimaryStyle, ReferenceAsset['style']> = {
  manga_anime: 'Manga / anime',
  western_ink: 'Western ink',
  realistic_academic: 'Realistic',
  cartoon: 'Cartoon',
  gesture_sketch: 'Gesture',
}

export function matchLabel(relevance: number): ReferenceAsset['match'] {
  if (relevance >= 0.8) return 'Strong'
  if (relevance >= 0.6) return 'Close'
  return 'Related'
}

function describe(assetId: string, scopes: SearchResult['scopes']): { title: string; scope: string } {
  const scope = scopes.map((label) => SCOPE_TITLES[label]).join(' · ') || 'Reference'
  return { title: `${scope} ${assetId.slice(-4)}`, scope }
}

export function toReferenceAsset(result: SearchResult): ReferenceAsset {
  const { title, scope } = describe(result.asset_id, result.scopes)
  return {
    id: result.asset_id,
    title,
    imageUrl: result.thumbnail_url,
    fullImageUrl: result.asset_url,
    style: API_STYLE_TO_UI[result.style],
    scope,
    source: 'Local gallery',
    // Presentation only: origin never decides what the asset may be used for.
    native: result.origin === 'native_line_art',
    match: matchLabel(result.relevance),
    relevance: result.relevance,
    // Permission comes from the stored source metadata, and the trace source
    // exists only when that permission does.
    traceAllowed: result.trace_allowed,
    traceUrl: result.trace_allowed ? result.asset_url : null,
  }
}

/** Same projection for a pinned asset, which the server revalidates on read. */
export function toPinnedReferenceAsset(pin: PinnedAsset): ReferenceAsset {
  const { title, scope } = describe(pin.asset_id, pin.scopes)
  return {
    id: pin.asset_id,
    title,
    imageUrl: pin.thumbnail_url,
    fullImageUrl: pin.asset_url,
    style: API_STYLE_TO_UI[pin.style],
    scope,
    source: 'Local gallery',
    native: pin.origin === 'native_line_art',
    match: 'Related',
    traceAllowed: pin.trace_allowed,
    traceUrl: pin.trace_allowed ? pin.asset_url : null,
    pinnedAt: pin.pinned_at,
  }
}

export function toPinSnapshot(response: PinsResponse): PinSnapshot {
  return {
    pins: response.pins.map(toPinnedReferenceAsset),
    revoked: (response.revoked ?? []).map((item) => ({ assetId: item.asset_id, reasons: item.reasons })),
  }
}

export function toReferenceGroup(group: SearchGroup): ReferenceGroup {
  return {
    id: group.kind === 'best_match' ? 'best' : group.id,
    title: group.kind === 'style' && group.style ? STYLE_TITLES[group.style] : group.title,
    tentative: group.kind === 'provisional_scope',
    results: group.results.map(toReferenceAsset),
  }
}

export function toSearchResponse(response: ApiSearchResponse, request: SearchRequest): SearchResponse {
  const top = response.scope_predictions[0]
  // The gallery owns the sufficiency verdict, and that verdict is measured on
  // raster ink. A blank canvas is a different *state* from "keep drawing" — it
  // is a property of what is on the canvas at all (nothing, or a fully
  // transparent import), which is why `blank_raster` maps to `empty` — but no
  // client-side count gets to override the answer: a vector-less import with
  // real ink in it has results to show.
  const blank = (response.degradations ?? []).some((item) => item.kind === 'blank_raster')
  const interpretation =
    response.mode === 'insufficient'
      ? blank
        ? 'Blank canvas'
        : 'Keep drawing'
      : top && top.label !== 'unknown'
        ? `${SCOPE_TITLES[top.label]} · ${Math.round(top.confidence * 100)}%`
        : 'Reading early marks'
  return {
    revision: response.revision,
    generation: request.generation,
    mode: response.mode === 'insufficient' && blank ? 'empty' : response.mode,
    interpretation,
    groups: response.groups.map(toReferenceGroup),
    warning: response.warning ?? null,
    degradations: response.degradations ?? [],
    timing: response.timing,
    // Provenance echoed from the server so the dock can disclose approximate
    // counts and the preprocessing/build that produced this result.
    countsApproximate: response.counts_approximate,
    strokeStatus: response.stroke_status,
    preprocessingVersion: response.preprocessing_version,
    apiVersion: response.api_version,
  }
}

export function toHealthResult(health: HealthResponse): HealthResult {
  const mode = health.fixture_mode ? 'fixture' : health.device === 'cuda' ? 'cuda' : 'cpu'
  const gallery = `${health.gallery_size.toLocaleString()} references`
  const message = !health.ready
    ? (health.warnings[0] ?? 'API is not ready')
    : health.device === 'cuda'
      ? `${health.gpu_name ?? 'GPU'} · ${gallery}`
      : `CPU fallback — slower search · ${gallery}`
  return { mode, ready: health.ready, message, live: true, health }
}

function toStyleSelection(selected: string | null): StyleSelection | undefined {
  if (!selected) return undefined
  return UI_STYLE_TO_API[selected] ?? undefined
}

export const liveServices: FrontendServices = {
  health: {
    get: async (signal) => toHealthResult(await api.health(signal)),
  },
  search: {
    async search(request, signal) {
      if (!request.image) throw new Error('Live search requires a snapshot image')
      const response = await api.search(
        {
          sessionId: request.sessionId,
          revision: request.revision,
          canvasWidth: LOGICAL_SIZE,
          canvasHeight: LOGICAL_SIZE,
          strokeCount: request.strokeCount,
          pointCount: request.pointCount,
          image: request.image,
          strokes: request.strokes,
          textHint: request.textHint || undefined,
          selectedStyle: toStyleSelection(request.selectedStyle),
        },
        signal,
      )
      return toSearchResponse(response, request)
    },
  },
  assets: {
    async resolveTrace(assetId: string, signal?: AbortSignal) {
      // Ask the server, never guess a URL: a saved document must not be able
      // to resurrect an asset whose trace permission (or eligibility) was
      // revoked since it was written.
      try {
        const permissions = await api.assetPermissions(assetId, signal)
        return permissions.allowed_trace ? (permissions.trace_url ?? null) : null
      } catch (error) {
        if (error instanceof ApiError) return null
        throw error
      }
    },
  },
  events: {
    record: (event) => api.recordEvent(event).then(() => undefined),
  },
  preferences: {
    get: () => api.getPreferences(),
    update: (update) => api.updatePreferences(update),
  },
  pins: {
    async list() {
      // Older builds cached live pins in localStorage. Hand those ids to the
      // API once so they become durable server state; ineligible ones are
      // simply dropped by the same revalidation every other pin gets.
      for (const assetId of takeLegacyLivePinIds()) {
        try {
          await api.pinAsset(assetId)
        } catch (error) {
          if (!(error instanceof ApiError)) throw error
        }
      }
      return toPinSnapshot(await api.getPins())
    },
    async pin(asset) {
      return toPinSnapshot(await api.pinAsset(asset.id))
    },
    async unpin(assetId) {
      return toPinSnapshot(await api.unpinAsset(assetId))
    },
  },
}
