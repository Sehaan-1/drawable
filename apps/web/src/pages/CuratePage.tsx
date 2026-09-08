import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { AlertCircle, FlaskConical, Inbox, Keyboard, RotateCw, X } from 'lucide-react'
import { ResearchShell } from '../components/AppChrome'
import { Button } from '../components/primitives'
import { useServiceStore } from '../services/serviceRegistry'
import {
  useAdjudicateSfw,
  useCreateCrop,
  useCurationNext,
  useCurationProgress,
  useExportSnapshot,
  useProcessDerivative,
  useQuarantine,
  useRevealQuarantined,
  useSkipCandidate,
  useWriteLabel,
} from '../services/curationHooks'
import { fetchCandidate } from '../services/curationClient'
import { CurateSidebar, type ScopeFilter, type StyleFilter } from '../components/CuratePage/CurateSidebar'
import { CurateStage } from '../components/CuratePage/CurateStage'
import { QuarantinePanel } from '../components/CuratePage/QuarantinePanel'
import {
  CurateInspector,
  type ReviewFormState,
} from '../components/CuratePage/CurateInspector'
import {
  defaultCrop,
  type PixelRect,
} from '../components/CuratePage/CropOverlay'
import type { CurationCandidate, LabelRequest } from '@drawable/contracts'
import { ApiError } from '../services/apiClient'
import {
  fixtureCurationCandidate,
  fixtureCurationProgress,
  fixtureQuarantine,
} from '../services/curationFixtures'

/**
 * Curation workspace.
 *
 * Wires the curation API (via React Query) to the sidebar, the candidate
 * image stage, the metadata inspector, and the SFW adjudication panel.
 *
 * Queue semantics: the page owns a per-mount review ``session_id``. The
 * server keeps a cursor for it — every serve advances, Skip excludes an
 * asset for the session (button or ``S``), and no local bookkeeping can
 * drift from what the queue actually serves. ``Previous`` re-fetches the
 * actual previous candidate **by id** (live metadata, never a stale copy).
 *
 * Concurrency: every write echoes the candidate's ``label_version`` back as
 * ``expected_label_version``. When another curator got there first, the API
 * answers 409 with reconciliation details and this page shows a banner —
 * the losing write is never applied, and one click reloads the live
 * candidate so the decision can be re-applied against the current version.
 *
 * Keyboard shortcuts are registered globally while the page is mounted so
 * the reviewer can drive the queue with one hand on the keyboard. The
 * handler ignores key events whose ``target`` is an editable element so
 * typing in the note textarea or selecting a chip does not get swallowed.
 */

const HISTORY_LIMIT = 50

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  if (target.isContentEditable) return true
  return false
}

/** Per-mount review session id (cursor/skip queue on the server). */
function newSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `sess-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`
}

interface ConflictBanner {
  assetId: string
  code: string
  message: string
  currentLabelVersion: number
  currentReviewState: string
  latestDecision: string | null
  latestReviewer: string | null
}

function conflictFromError(assetId: string, error: unknown): ConflictBanner | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null
  if (error.code !== 'label_version_conflict' && error.code !== 'review_conflict') return null
  const details = (error.details ?? {}) as Record<string, unknown>
  return {
    assetId,
    code: error.code,
    message: error.message,
    currentLabelVersion: typeof details.current_label_version === 'number' ? details.current_label_version : 0,
    currentReviewState: typeof details.current_review_state === 'string' ? details.current_review_state : 'unknown',
    latestDecision: typeof details.latest_decision === 'string' ? details.latest_decision : null,
    latestReviewer: typeof details.latest_reviewer === 'string' ? details.latest_reviewer : null,
  }
}

interface DerivativeToast {
  childId: string
  parentAssetId: string
  state: 'created' | 'processing' | 'complete' | 'failed' | 'cut_failed'
  detail?: string
}

/**
 * Build a human-facing detail for a failed crop cut. A 422
 * ``parent_artifact_invalid`` carries ``details.problems`` naming each parent
 * artifact that failed verification; show them so the curator knows the parent
 * needs a pipeline re-run rather than another crop attempt.
 */
function cropFailureDetail(error: unknown): string | undefined {
  if (!(error instanceof ApiError)) return undefined
  const problems = error.details?.problems
  if (Array.isArray(problems) && problems.length > 0) {
    return `${error.message}: ${problems.map(String).join('; ')}`
  }
  return error.message
}

export default function CuratePage() {
  const mode = useServiceStore((state) => state.mode)
  const health = useServiceStore((state) => state.health)
  const probe = useServiceStore((state) => state.probe)
  const queryClient = useQueryClient()

  // Make sure the registry has probed the API at least once so we know
  // whether the curation endpoints are reachable.
  useEffect(() => {
    if (mode === 'probing') void probe()
  }, [mode, probe])

  const curationEnabled = health?.health?.curation_enabled === true
  const live = mode === 'live'
  const offline = mode === 'fixture'

  // ---- session -----------------------------------------------------------
  // One session per mount: the server-side cursor makes Next/Skip advance
  // without a label, and skipped assets never return for this session.
  const sessionIdRef = useRef<string>(newSessionId())
  const sessionId = sessionIdRef.current

  // ---- filters ----------------------------------------------------------
  const [styleFilter, setStyleFilter] = useState<StyleFilter>(null)
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>(null)

  // ---- candidate fetch --------------------------------------------------
  const nextQuery = useCurationNext(
    {
      style: styleFilter ?? undefined,
      scope: scopeFilter ?? undefined,
      sessionId: live ? sessionId : undefined,
    },
    { enabled: live && curationEnabled },
  )

  // The currently *displayed* candidate. We never replace this with the
  // result of an in-flight ``useCurationNext`` unless we're idle (no mutation
  // in flight, no edit in progress); otherwise the UI would jump under the
  // reviewer's hands while they are mid-decision.
  const [current, setCurrent] = useState<CurationCandidate | null>(null)
  const [history, setHistory] = useState<string[]>([])

  const pushHistory = useCallback((assetId: string) => {
    setHistory((stack) => {
      if (stack[stack.length - 1] === assetId) return stack
      return [...stack.slice(-(HISTORY_LIMIT - 1)), assetId]
    })
  }, [])

  // When the query result changes and we're not busy, advance the displayed
  // candidate. We track the previous id on the history stack so Previous can
  // re-fetch it live by id.
  useEffect(() => {
    if (nextQuery.data === undefined) return // still loading or errored
    if (nextQuery.isFetching) return
    setCurrent((previous: CurationCandidate | null) => {
      if (previous && previous.asset_id === nextQuery.data?.asset_id) return previous
      if (previous && nextQuery.data) pushHistory(previous.asset_id)
      return nextQuery.data
    })
  }, [nextQuery.data, nextQuery.isFetching, pushHistory])

  const displayed = offline ? fixtureCurationCandidate : current

  // ---- form state -------------------------------------------------------
  const [form, setForm] = useState<ReviewFormState | null>(null)
  // Reset the form whenever the candidate changes.
  useEffect(() => {
    setForm(null)
  }, [displayed?.asset_id])

  // ---- crop editing -----------------------------------------------------
  const [editingCrop, setEditingCrop] = useState(false)
  const [crop, setCrop] = useState<PixelRect | null>(null)
  useEffect(() => {
    // Drop the previous crop when the candidate changes; the stage will
    // seed a fresh default once the new image's dimensions are known.
    setCrop(null)
    setEditingCrop(false)
  }, [displayed?.asset_id])

  // ---- conflict + toast state -------------------------------------------
  const [conflict, setConflict] = useState<ConflictBanner | null>(null)
  useEffect(() => {
    // The banner describes one asset; a candidate change supersedes it.
    if (conflict && displayed && conflict.assetId !== displayed.asset_id) setConflict(null)
  }, [conflict, displayed])
  const [derivativeToast, setDerivativeToast] = useState<DerivativeToast | null>(null)
  useEffect(() => {
    if (!derivativeToast) return
    const id = window.setTimeout(() => setDerivativeToast(null), 8000)
    return () => window.clearTimeout(id)
  }, [derivativeToast])

  // ---- label submission -------------------------------------------------
  const writeLabel = useWriteLabel()
  const exportSnapshot = useExportSnapshot()
  const skip = useSkipCandidate()
  const createCrop = useCreateCrop()
  const processDerivative = useProcessDerivative()
  const reveal = useRevealQuarantined()
  const adjudicate = useAdjudicateSfw()
  const busy =
    !offline &&
    (writeLabel.isPending || exportSnapshot.isPending || skip.isPending || createCrop.isPending)

  const reloadCandidateLive = useCallback(
    async (assetId: string): Promise<void> => {
      try {
        const fresh = await queryClient.fetchQuery({
          queryKey: ['curation', 'candidate', assetId],
          queryFn: () => fetchCandidate(assetId),
          staleTime: 0,
        })
        setCurrent(fresh)
      } catch {
        // The asset vanished (e.g. a gallery reload dropped it); keep the
        // stale copy on screen — the next /next serve will move on.
      }
    },
    [queryClient],
  )

  const submit = useCallback(
    (decision: 'keep' | 'reject') => {
      if (offline || !current || !form) return
      // Quality and a known primary scope are required by the API on keep,
      // and a blocked asset can only be rejected. We block here instead of
      // letting the server bounce with 422 to keep the UX snappy.
      if (
        decision === 'keep' &&
        (form.quality === null || form.primaryScope === 'unknown' || form.blockers.length > 0)
      ) {
        return
      }
      const payload: LabelRequest = {
        asset_id: current.asset_id,
        // Optimistic concurrency: echo the version we loaded. If another
        // curator wrote first, the server answers 409 and we reconcile.
        expected_review_state: current.review_state,
        expected_label_version: current.label_version,
        decision,
        primary_style: form.primaryStyle === current.primary_style ? null : form.primaryStyle,
        primary_scope: form.primaryScope === current.primary_scope ? null : form.primaryScope,
        secondary_scopes: sameScopes(form.secondaryScopes, current.secondary_scopes ?? [])
          ? null
          : form.secondaryScopes,
        blockers: form.blockers,
        quality: form.quality,
        note: form.note.trim() || null,
        sfw_safe: form.sfwSafe,
        session_id: sessionId,
      }
      writeLabel.mutate(payload, {
        onSuccess: () => {
          setConflict(null)
          // The mutation's onSuccess already invalidates the candidate +
          // progress caches, so a fresh /next will arrive shortly.
        },
        onError: (error) => {
          const banner = conflictFromError(current.asset_id, error)
          if (banner) setConflict(banner)
        },
      })
    },
    [offline, current, form, sessionId, writeLabel],
  )

  // ---- navigation -------------------------------------------------------
  const onPrev = useCallback(() => {
    if (offline) return
    setHistory((stack) => {
      if (stack.length === 0) return stack
      const previousId = stack[stack.length - 1]!
      // Re-fetch the *actual* previous candidate by id — live metadata, not
      // a stale copy — and show it immediately.
      void reloadCandidateLive(previousId)
      return stack.slice(0, -1)
    })
  }, [offline, reloadCandidateLive])

  const onNext = useCallback(() => {
    if (offline) return
    if (current) pushHistory(current.asset_id)
    void nextQuery.refetch()
  }, [offline, current, nextQuery, pushHistory])

  const onSkip = useCallback(() => {
    if (offline || !current) return
    if (current) pushHistory(current.asset_id)
    skip.mutate(
      { session_id: sessionId, asset_id: current.asset_id },
      {
        onSuccess: () => {
          setConflict(null)
          void nextQuery.refetch()
        },
      },
    )
  }, [offline, current, sessionId, skip, nextQuery, pushHistory])

  // ---- crop derivatives -------------------------------------------------
  // Committing a crop no longer edits the label: it cuts an immutable child
  // derivative (fresh files, its own processing + review state) and then
  // runs the required processing so it lands in the review queue.
  const onCreateDerivative = useCallback(() => {
    if (offline || !current || !crop) return
    createCrop.mutate(
      {
        assetId: current.asset_id,
        body: {
          crop: { x: crop.x, y: crop.y, width: crop.width, height: crop.height },
          expected_label_version: current.label_version,
          reviewer: null,
          note: null,
        },
      },
      {
        onSuccess: (created) => {
          setDerivativeToast({
            childId: created.asset_id,
            parentAssetId: current.asset_id,
            state: 'processing',
          })
          processDerivative.mutate(
            { assetId: created.asset_id },
            {
              onSuccess: (processed) => {
                setDerivativeToast({
                  childId: created.asset_id,
                  parentAssetId: current.asset_id,
                  state: processed.processing_state === 'complete' ? 'complete' : 'failed',
                  detail:
                    processed.processing_state === 'complete'
                      ? `quality ${(processed.measurements?.quality_score ?? 0).toFixed(2)} — awaiting its own review`
                      : undefined,
                })
              },
              onError: (error) => {
                setDerivativeToast({
                  childId: created.asset_id,
                  parentAssetId: current.asset_id,
                  state: 'failed',
                  detail: error instanceof ApiError ? error.message : undefined,
                })
              },
            },
          )
        },
        onError: (error) => {
          const banner = conflictFromError(current.asset_id, error)
          if (banner) {
            setConflict(banner)
            return
          }
          // The cut never happened: nothing about the parent was modified.
          setDerivativeToast({
            childId: current.asset_id,
            parentAssetId: current.asset_id,
            state: 'cut_failed',
            detail: cropFailureDetail(error),
          })
        },
      },
    )
  }, [offline, current, crop, createCrop, processDerivative])

  // ---- keyboard shortcuts ----------------------------------------------
  // We keep the latest ``submit`` and ``form`` in refs so the keydown
  // effect can be registered exactly once (no churn on every state
  // change) while still dispatching with the freshest values. The refs
  // are updated on every render but the listener subscription never
  // tears down.
  const submitRef = useRef(submit)
  useEffect(() => {
    submitRef.current = submit
  }, [submit])
  const formRef = useRef(form)
  useEffect(() => {
    formRef.current = form
  }, [form])
  const skipRef = useRef(onSkip)
  useEffect(() => {
    skipRef.current = onSkip
  }, [onSkip])

  // Tracks the last shortcut we acted on so the visible HUD can confirm
  // the listener is alive. ``lastKey`` is the key string, ``lastAt`` is a
  // monotonic counter we bump to force the auto-dismiss timer to reset
  // when the user mashes keys.
  const [lastKey, setLastKey] = useState<string | null>(null)
  const [lastAt, setLastAt] = useState(0)

  // Visible warning when the user presses K but no quality has been
  // selected yet. The keep shortcut stays disabled client-side, but
  // previously it silently no-oped, which made the keyboard feel broken.
  const [keepHint, setKeepHint] = useState<string | null>(null)

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (isEditableTarget(event.target)) return
      // Bail out of repeating events for everything except arrow navigation
      // so a held key can't spam the API.
      if (event.repeat && event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
      const fire = (key: string, action: () => void) => {
        event.preventDefault()
        setLastKey(key)
        setLastAt((n) => n + 1)
        action()
      }
      switch (event.key) {
        case 'k':
        case 'K':
          fire('K', () => {
            // Re-read form via the ref to avoid stale state. The submit
            // callback bails out cleanly when the keep preconditions are
            // not met, but we also surface a visible warning naming the
            // missing precondition so the keyboard doesn't *feel* broken.
            const liveForm = formRef.current
            const hint =
              liveForm === null
                ? 'nothing selected'
                : liveForm.quality === null
                  ? 'press 1, 2 or 3 to pick a quality'
                  : liveForm.primaryScope === 'unknown'
                    ? 'set a known primary scope first'
                    : liveForm.blockers.length > 0
                      ? 'clear the blockers or press R to reject'
                      : null
            if (hint === null) {
              submitRef.current('keep')
              setKeepHint(null)
            } else {
              setKeepHint(hint)
            }
          })
          break
        case 'r':
        case 'R':
          fire('R', () => submitRef.current('reject'))
          break
        case 's':
        case 'S':
          fire('S', () => skipRef.current())
          break
        case '1':
        case '2':
        case '3': {
          if (!displayed) return
          fire(event.key, () => {
            const quality = Number(event.key) as 1 | 2 | 3
            setForm((previous) => ({
              primaryStyle: displayed.primary_style,
              primaryScope: displayed.primary_scope,
              secondaryScopes: [...(displayed.secondary_scopes ?? [])],
              quality,
              note: previous?.note ?? '',
              blockers: previous?.blockers ?? [],
              sfwSafe: previous?.sfwSafe ?? null,
            }))
            setKeepHint(null)
          })
          break
        }
        case 'c':
        case 'C':
          fire('C', () => setEditingCrop((value) => !value))
          break
        case 'ArrowLeft':
          fire('←', onPrev)
          break
        case 'ArrowRight':
          fire('→', onNext)
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [displayed, onPrev, onNext])

  // Auto-dismiss the key indicator after 1.2s and the missing-quality
  // warning after 2s.
  useEffect(() => {
    if (lastKey === null) return
    const id = window.setTimeout(() => setLastKey(null), 1200)
    return () => window.clearTimeout(id)
  }, [lastAt, lastKey])
  useEffect(() => {
    if (keepHint === null) return
    const id = window.setTimeout(() => setKeepHint(null), 2000)
    return () => window.clearTimeout(id)
  }, [keepHint])

  // ---- progress + quarantine --------------------------------------------
  const progressQuery = useCurationProgress({ enabled: live && curationEnabled })
  const [quarantineMode, setQuarantineMode] = useState(false)
  const quarantineQuery = useQuarantine({ enabled: live && curationEnabled && quarantineMode })

  const onReveal = useCallback(
    (assetId: string) => {
      if (offline) return
      reveal.mutate({ assetId })
    },
    [offline, reveal],
  )

  const onAdjudicate = useCallback(
    (assetId: string, safe: boolean, expectedLabelVersion: number) => {
      if (offline) return
      adjudicate.mutate(
        { assetId, safe, expected_label_version: expectedLabelVersion },
        {
          onSuccess: () => {
            if (current?.asset_id === assetId) void reloadCandidateLive(assetId)
          },
        },
      )
    },
    [offline, adjudicate, current, reloadCandidateLive],
  )

  // ---- layout pieces ---------------------------------------------------
  const totalSeen = useMemo(() => {
    if (!displayed) return null
    if (offline) return { current: 1, total: 1 }
    return { current: history.length + 1, total: history.length + 1 }
  }, [displayed, history.length, offline])

  // The crop overlay's rect is staged locally; ``onCropCommit`` mirrors it
  // and the toolbar's "Create derivative" button is what persists it.
  const onCropChange = useCallback((next: PixelRect) => setCrop(next), [])
  const onCropCommit = useCallback((next: PixelRect) => setCrop(next), [])

  // Disabled / placeholder states ---------------------------------------
  if (mode === 'probing') {
    return (
      <ResearchShell eyebrow="Dataset workspace" title="Curation">
        <div className="curation-layout curation-layout--empty">
          <div className="curation-placeholder">
            <FlaskConical size={22} />
            <p>Connecting to the local API…</p>
          </div>
        </div>
      </ResearchShell>
    )
  }

  if (live && !curationEnabled) {
    return (
      <ResearchShell eyebrow="Dataset workspace" title="Curation">
        <div className="curation-layout curation-layout--empty">
          <div className="curation-placeholder">
            <AlertCircle size={22} />
            <h2>Curation mode is off</h2>
            <p>
              The API is reachable but the curation endpoints are not mounted. Set{' '}
              <code>LINESCOUT_CURATION_MODE=1</code> in <code>services/api/.env</code>{' '}
              and restart it.
            </p>
          </div>
        </div>
      </ResearchShell>
    )
  }

  if (!offline && nextQuery.isError && !quarantineMode) {
    return (
      <ResearchShell eyebrow="Dataset workspace" title="Curation">
        <div className="curation-layout curation-layout--empty">
          <div className="curation-placeholder">
            <AlertCircle size={22} />
            <h2>Could not reach the curation API</h2>
            <p>{nextQuery.error.message}</p>
            <Button onClick={() => void nextQuery.refetch()}>
              <RotateCw size={15} /> Retry
            </Button>
          </div>
        </div>
      </ResearchShell>
    )
  }

  const queueEmpty = !offline && current === null && !nextQuery.isLoading
  const pendingAsset =
    reveal.isPending || adjudicate.isPending
      ? (reveal.variables?.assetId ?? adjudicate.variables?.assetId ?? null)
      : null

  return (
    <ResearchShell eyebrow="Dataset workspace" title="Curation">
      <div className="curation-layout">
        <CurateSidebar
          progress={offline ? fixtureCurationProgress : progressQuery.data ?? null}
          style={styleFilter}
          scope={scopeFilter}
          onStyle={setStyleFilter}
          onScope={setScopeFilter}
          quarantined={offline ? fixtureQuarantine.length : progressQuery.data?.quarantined ?? 0}
          quarantineMode={quarantineMode}
          onToggleQuarantine={() => setQuarantineMode((value) => !value)}
        />

        {conflict ? (
          <div className="conflict-banner" role="alert" data-testid="conflict-banner">
            <AlertCircle size={15} />
            <div>
              <strong>
                {conflict.latestDecision
                  ? `Another curator already recorded “${conflict.latestDecision}”`
                  : 'This asset changed while you were reviewing it'}
                {conflict.latestReviewer ? ` (${conflict.latestReviewer})` : ''}
              </strong>
              <span>
                Live version is {conflict.currentLabelVersion} ({conflict.currentReviewState}).
                Your decision was not applied — reload the candidate and re-apply it.
              </span>
            </div>
            <Button onClick={() => void reloadCandidateLive(conflict.assetId)} data-testid="conflict-reload">
              <RotateCw size={13} /> Reload candidate
            </Button>
            <button
              type="button"
              className="kbd-toast__close"
              onClick={() => setConflict(null)}
              aria-label="Dismiss"
            >
              <X size={12} />
            </button>
          </div>
        ) : null}

        {derivativeToast ? (
          <div className="kbd-toast derivative-toast" role="status" data-testid="derivative-toast">
            {derivativeToast.state === 'complete' ? '✓' : derivativeToast.state === 'failed' || derivativeToast.state === 'cut_failed' ? '✕' : '…'}
            <span>
              {derivativeToast.state === 'cut_failed' ? (
                <>
                  Crop derivative of <code>{derivativeToast.parentAssetId}</code> was not cut
                  {derivativeToast.detail ? `: ${derivativeToast.detail}` : ''} — the parent was
                  not modified.
                </>
              ) : (
                <>
                  Derivative <code>{derivativeToast.childId}</code> cut from{' '}
                  <code>{derivativeToast.parentAssetId}</code>
                  {derivativeToast.state === 'processing' ? ' — processing…' : ''}
                  {derivativeToast.state === 'complete' ? ` — ${derivativeToast.detail ?? 'processed'}` : ''}
                  {derivativeToast.state === 'failed'
                    ? ` — processing failed${derivativeToast.detail ? `: ${derivativeToast.detail}` : ''} (retry from the queue)`
                    : ''}
                </>
              )}
            </span>
            <button
              type="button"
              className="kbd-toast__close"
              onClick={() => setDerivativeToast(null)}
              aria-label="Dismiss"
            >
              <X size={12} />
            </button>
          </div>
        ) : null}

        {quarantineMode && !offline ? (
          <QuarantinePanel
            entries={quarantineQuery.data ?? null}
            loading={quarantineQuery.isLoading}
            error={quarantineQuery.error ? quarantineQuery.error.message : null}
            onRetry={() => void quarantineQuery.refetch()}
            onReveal={onReveal}
            onAdjudicate={onAdjudicate}
            pendingAsset={pendingAsset}
          />
        ) : queueEmpty ? (
          <section className="candidate-stage candidate-stage--empty">
            <div className="candidate-empty-card">
              <Inbox size={32} />
              <h2>Queue empty</h2>
              <p>
                Every candidate matching this filter has been reviewed. Try
                a different scope or style, or export a snapshot of the
                work so far.
              </p>
              <Button
                onClick={() => exportSnapshot.mutate()}
                disabled={exportSnapshot.isPending}
              >
                {exportSnapshot.isPending ? 'Exporting…' : 'Export snapshot'}
              </Button>
              {exportSnapshot.isSuccess ? (
                <p className="snapshot-result" data-testid="snapshot-result">
                  Wrote <code>{exportSnapshot.data.path}</code> with{' '}
                  {exportSnapshot.data.label_count} labels
                  (snapshot {exportSnapshot.data.snapshot_id}).
                </p>
              ) : null}
            </div>
          </section>
        ) : (
          <CurateStage
            candidate={displayed}
            editingCrop={editingCrop}
            onToggleCrop={() => setEditingCrop((value) => !value)}
            crop={crop}
            onCropChange={onCropChange}
            onCropCommit={onCropCommit}
            onPrev={onPrev}
            onNext={onNext}
            onSkip={offline ? undefined : onSkip}
            onReset={() => {
              if (displayed) {
                const next = defaultCrop(displayed.width, displayed.height)
                setCrop(next)
              }
            }}
            onCreateDerivative={offline ? undefined : onCreateDerivative}
            derivativePending={createCrop.isPending || processDerivative.isPending}
            hasPrev={!offline && history.length > 0}
            hasNext={!offline}
            position={totalSeen}
            busy={busy}
          />
        )}

        <CurateInspector
          candidate={displayed}
          pendingForm={form}
          onFormChange={setForm}
          onKeep={() => submit('keep')}
          onReject={() => submit('reject')}
          onSnapshot={() => {
            if (!offline) exportSnapshot.mutate()
          }}
          busy={busy}
          snapshotPending={!offline && exportSnapshot.isPending}
          disabled={offline || !curationEnabled}
        />

        {/* Keyboard shortcut HUD. Floats over the layout so the user can
            see what each key does and confirm the listener is firing. */}
        <aside
          className="kbd-hud"
          aria-label="Keyboard shortcuts"
          data-testid="kbd-hud"
        >
          <header>
            <Keyboard size={13} />
            <span>Shortcuts</span>
            {lastKey ? (
              <span className="kbd-hud__pulse" data-testid="kbd-hud-last">
                {lastKey}
              </span>
            ) : null}
          </header>
          <dl>
            <div>
              <dt><kbd>K</kbd></dt>
              <dd>Keep current candidate</dd>
            </div>
            <div>
              <dt><kbd>R</kbd></dt>
              <dd>Reject current candidate</dd>
            </div>
            <div>
              <dt><kbd>S</kbd></dt>
              <dd>Skip without a label</dd>
            </div>
            <div>
              <dt><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd></dt>
              <dd>Set quality (required for Keep)</dd>
            </div>
            <div>
              <dt><kbd>C</kbd></dt>
              <dd>Toggle crop edit</dd>
            </div>
            <div>
              <dt><kbd>←</kbd> <kbd>→</kbd></dt>
              <dd>Previous / next candidate</dd>
            </div>
          </dl>
        </aside>

        {keepHint ? (
          <div
            className="kbd-toast"
            role="status"
            data-testid="kbd-toast-missing-quality"
          >
            <AlertCircle size={14} />
            <span>Can’t keep yet — {keepHint}.</span>
            <button
              type="button"
              className="kbd-toast__close"
              onClick={() => setKeepHint(null)}
              aria-label="Dismiss"
            >
              <X size={12} />
            </button>
          </div>
        ) : null}
      </div>
    </ResearchShell>
  )
}

function sameScopes(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  const sortedA = [...a].sort()
  const sortedB = [...b].sort()
  return sortedA.every((value, index) => value === sortedB[index])
}
