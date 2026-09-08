/**
 * Typed fetch helpers for the `/api/v1/curation/*` endpoints.
 *
 * The shared `apiClient` already handles the URL prefix, JSON encoding, and
 * structured error parsing; this module adds the curation-specific shapes
 * (which are generated into `@drawable/contracts`) and a couple of ergonomic
 * defaults — for example, a `GET /curation/next` 404 with code
 * ``queue_empty`` is normalised to a ``null`` return value so the React
 * Query layer can treat "no more candidates" as a normal state instead of
 * an error.
 *
 * Curation-safety semantics (schema v4) live here too: the session-scoped
 * cursor queue (``session_id`` + skip), stable by-id retrieval, SFW
 * adjudication behind reveal grants, and the crop-derivative lifecycle
 * (create → process). Every write sends the caller's last observed
 * ``expected_label_version``; a stale version surfaces as a 409 whose
 * ``details`` carry reconciliation information.
 */

import {
  type CurationCandidate,
  type CurationProgress,
  type CropRequest,
  type CropResponse,
  type DerivativeProcessResponse,
  type LabelRequest,
  type LabelResponse,
  type PrimaryStyle,
  type QuarantineCandidate,
  type RevealRequest,
  type ScopeLabel,
  type SfwAdjudicationRequest,
  type SfwAdjudicationResponse,
  type SkipRequest,
  type SkipResponse,
  type SnapshotResponse,
} from '@drawable/contracts'
import { ApiError, apiRequest as request } from './apiClient'

export interface NextCandidateQuery {
  style?: PrimaryStyle
  scope?: ScopeLabel
  /** Review session id: enables the cursor/skip queue semantics. */
  sessionId?: string
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

/** 404 -> ``null``, anything else propagates as an :class:`ApiError`. */
export async function fetchNextCandidate(
  query: NextCandidateQuery,
  signal?: AbortSignal,
): Promise<CurationCandidate | null> {
  const params = new URLSearchParams()
  if (query.style) params.set('style', query.style)
  if (query.scope) params.set('scope', query.scope)
  if (query.sessionId) params.set('session_id', query.sessionId)
  const path = params.toString() ? `/curation/next?${params}` : '/curation/next'
  try {
    return await request<CurationCandidate>(path, { signal })
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      // ``queue_empty`` is the "we drained the queue" path; ``gallery_unavailable``
      // means the API was started without a gallery manifest. Both are
      // first-class empty states from the UI's perspective.
      if (error.code === 'queue_empty' || error.code === 'gallery_unavailable') return null
    }
    throw error
  }
}

/**
 * Stable retrieval of one candidate by id, whatever its review state.
 *
 * This is the read path for Previous (re-fetch the *actual* previous
 * candidate — live metadata, never a stale copy) and for reconciliation
 * after a 409 label-version conflict.
 */
export async function fetchCandidate(
  assetId: string,
  signal?: AbortSignal,
): Promise<CurationCandidate> {
  return request<CurationCandidate>(`/curation/candidates/${encodeURIComponent(assetId)}`, {
    signal,
  })
}

export function fetchCurationProgress(signal?: AbortSignal): Promise<CurationProgress> {
  return request<CurationProgress>('/curation/progress', { signal })
}

export function writeLabel(body: LabelRequest): Promise<LabelResponse> {
  return post<LabelResponse>('/curation/labels', body)
}

/** Skip an asset for this session: the cursor advances, the asset never returns. */
export function skipCandidate(body: SkipRequest): Promise<SkipResponse> {
  return post<SkipResponse>('/curation/queue/skip', body)
}

/** The SFW adjudication backlog: held records, metadata only (no images). */
export function fetchQuarantine(signal?: AbortSignal): Promise<QuarantineCandidate[]> {
  return request<QuarantineCandidate[]>('/curation/quarantine', { signal })
}

/**
 * Issue a deliberate, expiring reveal grant for one held record.
 *
 * The grant opens the *curation preview* only — never the public asset
 * routes — and changes nothing about the record itself.
 */
export function revealQuarantined(
  assetId: string,
  body: RevealRequest = {},
): Promise<QuarantineCandidate> {
  return post<QuarantineCandidate>(`/curation/quarantine/${encodeURIComponent(assetId)}/reveal`, body)
}

/** Record an explicit human SFW adjudication (safe/unsafe) for one record. */
export function adjudicateSfw(
  assetId: string,
  body: SfwAdjudicationRequest,
): Promise<SfwAdjudicationResponse> {
  return post<SfwAdjudicationResponse>(
    `/curation/sfw/${encodeURIComponent(assetId)}/adjudication`,
    body,
  )
}

/**
 * Cut an immutable child derivative from a parent asset.
 *
 * The child gets fresh files/hashes, its own processing state (``pending``),
 * and its own review state (``unreviewed``) — nothing about the parent's
 * review, SFW, or gold status is inherited.
 */
export function createCrop(assetId: string, body: CropRequest): Promise<CropResponse> {
  return post<CropResponse>(`/curation/assets/${encodeURIComponent(assetId)}/crops`, body)
}

/** Run (or re-run) a derivative's required processing. */
export function processDerivative(assetId: string): Promise<DerivativeProcessResponse> {
  return post<DerivativeProcessResponse>(`/curation/assets/${encodeURIComponent(assetId)}/process`)
}

export function exportSnapshot(): Promise<SnapshotResponse> {
  return post<SnapshotResponse>('/curation/snapshots')
}
