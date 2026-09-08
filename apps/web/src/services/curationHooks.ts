/**
 * React Query bindings for the curation endpoints.
 *
 * The CuratePage composes these hooks; this module deliberately stays UI-free
 * so the same hooks can back tests, future bulk-review tools, or scripted
 * automation without a React tree.
 *
 * Query keys are kept stable: a re-render of the page must not invalidate a
 * candidate fetch simply because of an object identity change on the filter
 * inputs. We use a small ``stableQuery`` helper that joins only the fields
 * the API actually filters on.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from '@tanstack/react-query'
import type {
  CurationCandidate,
  CurationProgress,
  CropRequest,
  CropResponse,
  DerivativeProcessResponse,
  LabelRequest,
  LabelResponse,
  QuarantineCandidate,
  SfwAdjudicationRequest,
  SfwAdjudicationResponse,
  SkipRequest,
  SkipResponse,
  SnapshotResponse,
} from '@drawable/contracts'
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
  type NextCandidateQuery,
} from './curationClient'

const queryKeys = {
  progress: ['curation', 'progress'] as const,
  next: (query: NextCandidateQuery) =>
    [
      'curation',
      'next',
      query.style ?? null,
      query.scope ?? null,
      query.sessionId ?? null,
    ] as const,
  candidate: (assetId: string) => ['curation', 'candidate', assetId] as const,
  quarantine: ['curation', 'quarantine'] as const,
}

function stableQuery(input: NextCandidateQuery): NextCandidateQuery {
  return { style: input.style, scope: input.scope, sessionId: input.sessionId }
}

/**
 * The currently displayed candidate. ``null`` means "queue empty" or
 * "no gallery" — both are first-class UI states, not errors.
 *
 * With a ``sessionId`` the queue is cursor-based on the server: every serve
 * advances the session cursor, so a plain refetch yields the *next*
 * candidate without any local bookkeeping.
 */
export function useCurationNext(
  query: NextCandidateQuery,
  options: { enabled?: boolean } = {},
): UseQueryResult<CurationCandidate | null, Error> {
  return useQuery({
    queryKey: queryKeys.next(stableQuery(query)),
    queryFn: ({ signal }) => fetchNextCandidate(stableQuery(query), signal),
    enabled: options.enabled ?? true,
  })
}

/**
 * Live candidate by id — the read path for Previous (re-fetch the actual
 * previous candidate instead of trusting a stale copy) and for conflict
 * reconciliation after a 409.
 */
export function useCandidate(
  assetId: string | null,
  options: { enabled?: boolean } = {},
): UseQueryResult<CurationCandidate, Error> {
  return useQuery({
    queryKey: queryKeys.candidate(assetId ?? 'none'),
    queryFn: ({ signal }) => fetchCandidate(assetId as string, signal),
    enabled: (options.enabled ?? true) && assetId !== null,
  })
}

/** Review progress with per-style / per-scope breakdowns. */
export function useCurationProgress(options: { enabled?: boolean } = {}): UseQueryResult<CurationProgress, Error> {
  return useQuery({
    queryKey: queryKeys.progress,
    queryFn: ({ signal }) => fetchCurationProgress(signal),
    // Progress moves while the curator is working, so keep it live: refetch
    // every 5 seconds while the tab is visible. Refetch on focus is left to
    // React Query's default (off, set in main.tsx) — curators usually keep
    // the tab focused.
    refetchInterval: 5_000,
    enabled: options.enabled ?? true,
  })
}

/**
 * The SFW adjudication backlog: held records, metadata only. The images
 * stay hidden until a deliberate reveal grant exists.
 */
export function useQuarantine(options: { enabled?: boolean } = {}): UseQueryResult<QuarantineCandidate[], Error> {
  return useQuery({
    queryKey: queryKeys.quarantine,
    queryFn: ({ signal }) => fetchQuarantine(signal),
    enabled: options.enabled ?? true,
  })
}

/** Write a label. On success, invalidate progress and any cached candidate. */
export function useWriteLabel(): UseMutationResult<LabelResponse, Error, LabelRequest> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: writeLabel,
    onSuccess: (_data, variables) => {
      void client.invalidateQueries({ queryKey: queryKeys.progress })
      // Drop the cached "next" candidate: the queue advanced. Also refresh
      // the by-id view of the asset we just labelled.
      void client.invalidateQueries({ queryKey: ['curation', 'next'] })
      void client.invalidateQueries({ queryKey: queryKeys.candidate(variables.asset_id) })
    },
  })
}

/**
 * Skip the current candidate for this session: the queue advances without a
 * label and the asset never returns (for this session).
 */
export function useSkipCandidate(): UseMutationResult<SkipResponse, Error, SkipRequest> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: skipCandidate,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['curation', 'next'] })
    },
  })
}

/** Issue a deliberate, expiring reveal grant for one held record. */
export function useRevealQuarantined(): UseMutationResult<QuarantineCandidate, Error, { assetId: string; reviewer?: string | null }> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ assetId, reviewer }) => revealQuarantined(assetId, { reviewer: reviewer ?? null }),
    onSuccess: (updated) => {
      // Refetch the whole backlog so every entry's ``revealed`` flag is fresh.
      void client.invalidateQueries({ queryKey: queryKeys.quarantine })
      void client.setQueryData<QuarantineCandidate[]>(queryKeys.quarantine, (previous) =>
        previous?.map((entry) => (entry.asset_id === updated.asset_id ? updated : entry)),
      )
    },
  })
}

/**
 * Record an explicit human SFW adjudication. ``safe`` returns a quarantined
 * record to the review queue on its merits; ``unsafe`` quarantines it and
 * closes every serving surface until a later adjudication.
 */
export function useAdjudicateSfw(): UseMutationResult<
  SfwAdjudicationResponse,
  Error,
  SfwAdjudicationRequest & { assetId: string }
> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ assetId, ...body }) => adjudicateSfw(assetId, body),
    onSuccess: (_data, variables) => {
      void client.invalidateQueries({ queryKey: queryKeys.quarantine })
      void client.invalidateQueries({ queryKey: queryKeys.progress })
      void client.invalidateQueries({ queryKey: ['curation', 'next'] })
      void client.invalidateQueries({ queryKey: queryKeys.candidate(variables.assetId) })
    },
  })
}

/**
 * Cut an immutable crop derivative from the current candidate. The child is
 * created ``pending`` its own processing; chain :func:`useProcessDerivative`
 * (the page does) to complete the lifecycle.
 */
export function useCreateCrop(): UseMutationResult<
  CropResponse,
  Error,
  { assetId: string; body: CropRequest }
> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ assetId, body }) => createCrop(assetId, body),
    onSuccess: (created) => {
      // The child is retrievable by id immediately (pending processing).
      void client.setQueryData(queryKeys.candidate(created.asset_id), null)
      void client.invalidateQueries({ queryKey: queryKeys.candidate(created.asset_id) })
    },
  })
}

/** Run (or re-run) a derivative's required processing. */
export function useProcessDerivative(): UseMutationResult<
  DerivativeProcessResponse,
  Error,
  { assetId: string }
> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ assetId }) => processDerivative(assetId),
    onSuccess: (processed) => {
      void client.invalidateQueries({ queryKey: queryKeys.candidate(processed.asset_id) })
      void client.invalidateQueries({ queryKey: ['curation', 'next'] })
      void client.invalidateQueries({ queryKey: queryKeys.progress })
    },
  })
}

/** Export an immutable JSON snapshot. */
export function useExportSnapshot(): UseMutationResult<SnapshotResponse, Error, void> {
  const client = useQueryClient()
  return useMutation({
    mutationFn: exportSnapshot,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.progress })
    },
  })
}
