import { useEffect, useRef, useState } from 'react'
import { buildStrokeSequence, documentCounts } from '../lib/strokeSequence'
import { inputDegradations, rasterSufficiency } from '../lib/rasterInk'
import type { DrawingDocument } from '../lib/types'
import { useDocumentStore } from '../state/documentStore'
import { useSearchStore, type SearchOwnership } from '../state/searchStore'
import { useUiStore } from '../state/uiStore'
import { fixtureServices, type FrontendServices } from './frontendServices'
import { cleanupExpiredImports, loadDocument, materializeStagedImport, saveDocument, type LoadedDocument } from './persistence'
import { useServiceStore } from './serviceRegistry'
import { prepareSnapshot, type PreparedSnapshot } from './snapshotClient'
import { unavailableArtworkSummary } from './rasterAssets'
import { acquireDocumentLease, type DocumentLease } from './documentLock'

/** Idle time between a finished stroke and the next snapshot+search. */
const SEARCH_DEBOUNCE_MS = 350

function isAbort(error: unknown) {
  return error instanceof DOMException && error.name === 'AbortError'
}

/**
 * A snapshot is only usable by the request it was prepared for. The worker
 * echoes the whole ownership identity, so a snapshot that does not match —
 * different document *or* revision *or* generation *or* claim token — is
 * discarded instead of feeding a search nobody asked for any more.
 */
function answers(snapshot: PreparedSnapshot, ownership: SearchOwnership) {
  return (
    snapshot.token === ownership.token &&
    snapshot.documentId === ownership.documentId &&
    snapshot.revision === ownership.revision &&
    snapshot.generation === ownership.generation
  )
}

interface SearchRun {
  document: DrawingDocument
  ownership: SearchOwnership
  services: FrontendServices
  sessionId: string
  textHint: string
  selectedStyle: string | null
  signal: AbortSignal
}

/**
 * One search, end to end, under a single ownership token.
 *
 * Every exit writes through the token: a response that lost ownership is
 * dropped rather than applied, an error that lost ownership is not surfaced,
 * and the spinner is settled only by whoever still owns it. That is what keeps
 * a superseded request from clearing a newer request's loading state.
 */
async function runSearch({ document, ownership, services, sessionId, textHint, selectedStyle, signal }: SearchRun) {
  const store = useSearchStore.getState()
  try {
    const snapshot = await prepareSnapshot(document, ownership, signal)
    if (!answers(snapshot, ownership)) return
    // An import whose stored blob is gone is a degraded input, never a veto:
    // the search runs on the surviving ink and the omission travels on the
    // response's `warning`. It deliberately does *not* become a `degradations`
    // kind — that list is the gallery's attestation about a query it ranked,
    // and the gallery knows nothing about this device's storage.
    const omission = snapshot.unavailable.length
      ? `${unavailableArtworkSummary(snapshot.unavailable)}; the search used the surviving visible marks.`
      : null
    const counts = documentCounts(document.layers)
    const sequence = buildStrokeSequence(document.layers)
    // An empty vector payload is not sent at all: `stroke_status` has to mean
    // "the stroke branch had input", and a 0-stroke payload under a nonzero
    // stroke_count is exactly the mismatch the API rejects with a 422.
    const strokes = sequence.strokes.length ? sequence : undefined
    // Overall sufficiency is a property of the *ink*, so this is the one case
    // worth short-circuiting locally: a canvas with nothing visible on it
    // (cleared layers, or an import whose pixels are all transparent) has
    // nothing for any service to rank, and the vector counts say nothing about
    // whether that is true. How *little* ink is enough stays the gallery's
    // call, so anything with ink in it is still sent.
    if (rasterSufficiency(snapshot.ink).blank) {
      const degradations = inputDegradations(snapshot.ink, counts.strokeCount, counts.pointCount, strokes ? counts.pointCount : undefined)
      const details = degradations.map((item) => item.detail)
      if (omission) details.push(omission)
      store.resolve(ownership, {
        revision: ownership.revision,
        generation: ownership.generation,
        mode: 'empty',
        interpretation: 'Blank canvas',
        groups: [],
        degradations,
        warning: details.join('; ') || null,
        countsApproximate: strokes === undefined,
        strokeStatus: strokes ? 'present' : 'absent',
      })
      return
    }
    const response = await services.search.search({
      sessionId,
      revision: ownership.revision,
      generation: ownership.generation,
      strokeCount: counts.strokeCount,
      pointCount: counts.pointCount,
      rasterCount: counts.rasterCount,
      ink: snapshot.ink,
      textHint,
      selectedStyle,
      image: snapshot.image,
      strokes,
    }, signal)
    store.resolve(ownership, omission
      ? { ...response, warning: [response.warning, omission].filter((part): part is string => Boolean(part)).join('; ') }
      : response)
  } catch (error) {
    // An abort is the expected end of a superseded request; the effect cleanup
    // has already settled loading for it.
    if (isAbort(error) || signal.aborted) return
    store.reject(ownership, error instanceof Error ? error.message : 'Reference search failed.')
  }
}

export function useWorkspaceLifecycle() {
  const document = useDocumentStore((state) => state.document)
  const hasHydrated = useDocumentStore((state) => state.hasHydrated)
  const setHydrated = useDocumentStore((state) => state.setHydrated)
  const replaceDocument = useDocumentStore((state) => state.replaceDocument)
  const activeLayerId = useDocumentStore((state) => state.activeLayerId)
  const setResolvedTraceImage = useDocumentStore((state) => state.setResolvedTraceImage)
  const clearTrace = useDocumentStore((state) => state.clearTrace)
  const generation = useSearchStore((state) => state.generation)
  const drawing = useSearchStore((state) => state.drawing)
  const textHint = useSearchStore((state) => state.textHint)
  const selectedStyle = useSearchStore((state) => state.selectedStyle)
  const invalidate = useSearchStore((state) => state.invalidate)
  const claim = useSearchStore((state) => state.claim)
  const theme = useUiStore((state) => state.theme)
  const serviceMode = useServiceStore((state) => state.mode)
  const services = useServiceStore((state) => state.services)
  const sessionId = useServiceStore((state) => state.sessionId)
  const probeServices = useServiceStore((state) => state.probe)
  const [restoreCandidate, setRestoreCandidate] = useState<LoadedDocument | null>(null)
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'error'>('saved')
  const [notice, setNotice] = useState<string | null>(null)
  // Keyed on the *document*, not just its revision: switching to a different
  // document whose revision happens to be equal must still invalidate the
  // results of the one it replaced.
  const previousIdentity = useRef(`${document.id}:${document.revision}`)
  const lease = useRef<DocumentLease | null>(null)
  const activationVersion = useRef(0)

  const setDocumentUrl = (documentId: string) => {
    const url = new URL(window.location.href)
    url.search = ''
    url.searchParams.set('document', documentId)
    window.history.replaceState(null, '', `${url.pathname}${url.search}`)
  }

  const activateDocument = async (loaded: LoadedDocument) => {
    const activation = ++activationVersion.current
    // A restored document has to re-earn its trace layer: the cached image URL
    // is dropped so the effect below revalidates the asset against the current
    // gallery. A trace permission revoked since the save must not survive a
    // reload just because the URL was written into local storage.
    let next: LoadedDocument = {
      ...loaded,
      document: { ...loaded.document, trace: { ...loaded.document.trace, imageUrl: null } },
    }
    let nextLease = await acquireDocumentLease(next.document.id)
    if (activation !== activationVersion.current) {
      nextLease.release()
      return
    }
    if (!nextLease.acquired) {
      nextLease.release()
      next = {
        ...next,
        document: { ...structuredClone(next.document), id: `document-${crypto.randomUUID()}`, updatedAt: Date.now() },
      }
      nextLease = await acquireDocumentLease(next.document.id)
      setNotice('This drawing was already open, so drawable created an independent copy.')
    }
    if (activation !== activationVersion.current) {
      nextLease.release()
      return
    }
    lease.current?.release()
    lease.current = nextLease
    replaceDocument(next.document, next.activeLayerId)
    setDocumentUrl(next.document.id)
    setHydrated(true)
  }

  useEffect(() => { void probeServices() }, [probeServices])

  useEffect(() => {
    let active = true
    const load = async () => {
      void cleanupExpiredImports().catch(() => undefined)
      const parameters = new URLSearchParams(window.location.search)
      const importToken = parameters.get('import')
      const requestedDocument = parameters.get('document') ?? undefined
      const saved = importToken ? await materializeStagedImport(importToken) : await loadDocument(requestedDocument)
      if (!active) return
      if (importToken && !saved) {
        setNotice('This import link has expired. Return to the original tab and choose the file again.')
        const blank = useDocumentStore.getState().document
        await activateDocument({ document: blank, activeLayerId: 'layer-1' })
        return
      }
      const hasInk = saved?.document.layers.some((layer) => layer.operations.length)
      if (saved && hasInk && !importToken) setRestoreCandidate(saved)
      else if (saved) {
        await activateDocument(saved)
        if (importToken) setNotice('Imported sketch opened as a new local drawing.')
      } else {
        const blank = useDocumentStore.getState().document
        await activateDocument({ document: blank, activeLayerId: 'layer-1' })
      }
    }
    void load().catch(() => {
      if (active) {
        setNotice('Local recovery storage could not be opened. Drawing remains available, but autosave may be unavailable.')
        setHydrated(true)
      }
    })
    return () => { active = false; activationVersion.current += 1; lease.current?.release(); lease.current = null }
    // Startup must run once per mounted workspace; store changes are handled by the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setHydrated])

  useEffect(() => {
    if (!hasHydrated) return
    setSaveState('saving')
    const timer = window.setTimeout(() => {
      saveDocument(document, activeLayerId).then(() => setSaveState('saved')).catch(() => setSaveState('error'))
    }, 500)
    return () => window.clearTimeout(timer)
  }, [activeLayerId, document, hasHydrated])

  useEffect(() => {
    if (!hasHydrated || !document.trace.assetId || document.trace.imageUrl) return
    const controller = new AbortController()
    const assetClient = services.assets ?? fixtureServices.assets
    assetClient.resolveTrace(document.trace.assetId, controller.signal).then((imageUrl) => {
      if (imageUrl) {
        setResolvedTraceImage(imageUrl)
        return
      }
      // Gone, or no longer traceable: drop the reference entirely rather than
      // leave a trace layer pointing at an asset we may not use.
      clearTrace()
      setNotice('The saved trace reference is unavailable; the drawing opened without it.')
    }).catch(() => setNotice('The saved trace reference is unavailable; the drawing opened without it.'))
    return () => controller.abort()
  }, [clearTrace, document.trace.assetId, document.trace.imageUrl, hasHydrated, services, setResolvedTraceImage])

  useEffect(() => {
    const identity = `${document.id}:${document.revision}`
    if (previousIdentity.current === identity) return
    previousIdentity.current = identity
    // `invalidate` bumps the generation *and* drops the in-flight request with
    // it, so replacing a document can neither keep the old results nor keep the
    // old spinner alive on the new one.
    invalidate(false)
  }, [document.id, document.revision, invalidate])

  useEffect(() => {
    if (drawing || !hasHydrated || serviceMode === 'probing') return
    const controller = new AbortController()
    // The claim ties this run to the exact drawing state it searches. A mode
    // switch, a document swap, or a new stroke invalidates it; nothing this
    // effect started may write state after that.
    const ownership = claim({ documentId: document.id, revision: document.revision, generation })
    const timer = window.setTimeout(() => {
      const search = useSearchStore.getState()
      // Superseded during the debounce (another claim, an invalidate, a mode
      // switch): stay quiet rather than raise a spinner for a dead request.
      if (!search.isOwner(ownership)) return
      // Read the document from the store instead of closing over it, and check
      // it against the claim: a snapshot must be of the state the ownership
      // token was issued for, not of whatever render last ran this effect.
      const current = useDocumentStore.getState().document
      if (current.id !== ownership.documentId || current.revision !== ownership.revision) return
      search.begin(ownership)
      void runSearch({ document: current, ownership, services, sessionId, textHint, selectedStyle, signal: controller.signal })
    }, SEARCH_DEBOUNCE_MS)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
      // Settle the spinner this run raised — but only if nobody else has taken
      // ownership since. Unmount and "restart the search" both land here.
      useSearchStore.getState().release(ownership)
    }
  }, [claim, document.id, document.revision, drawing, generation, hasHydrated, selectedStyle, serviceMode, services, sessionId, textHint])

  useEffect(() => {
    if (theme !== 'system') return
    const query = matchMedia('(prefers-color-scheme: dark)')
    const apply = () => { globalThis.document.documentElement.dataset.theme = query.matches ? 'dark' : 'light' }
    query.addEventListener('change', apply)
    return () => query.removeEventListener('change', apply)
  }, [theme])

  const restore = () => {
    if (restoreCandidate) void activateDocument(restoreCandidate)
    setRestoreCandidate(null)
  }
  const discard = () => {
    setRestoreCandidate(null)
    const blank = useDocumentStore.getState().document
    void activateDocument({ document: blank, activeLayerId: 'layer-1' })
  }

  return { saveState, restoreCandidate: restoreCandidate?.document ?? null, restore, discard, notice, dismissNotice: () => setNotice(null) }
}
