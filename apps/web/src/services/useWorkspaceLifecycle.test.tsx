import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DrawingDocument, SearchRequest, SearchResponse } from '../lib/types'
import type { SearchOwnership } from '../state/searchStore'
import { useDocumentStore } from '../state/documentStore'
import { useSearchStore } from '../state/searchStore'
import { useUiStore } from '../state/uiStore'
import { useWorkspaceLifecycle } from './useWorkspaceLifecycle'

/**
 * Search-lifecycle races, driven deterministically.
 *
 * The snapshot client and the search client hand back deferred promises that
 * the test settles exactly when it wants to, so "who won the race" is a choice
 * rather than a timing accident. Everything here pins the ownership rule: a
 * request owns the results panel for one (document id, revision, generation)
 * triple plus its claim token, and a superseded request may neither deliver a
 * response, nor raise an error, nor settle the spinner of the request that
 * replaced it.
 */

const clients = vi.hoisted(() => ({ prepare: vi.fn(), search: vi.fn() }))

vi.mock('./snapshotClient', () => ({ prepareSnapshot: (...args: unknown[]) => clients.prepare(...args) }))

vi.mock('./persistence', () => ({
  cleanupExpiredImports: async () => 0,
  loadDocument: async () => null,
  materializeStagedImport: async () => null,
  saveDocument: async () => undefined,
  loadRasterAsset: async () => null,
  loadReferencedRasterAssets: async () => [],
}))

vi.mock('./documentLock', () => ({
  acquireDocumentLease: async () => ({ acquired: true, release: () => undefined }),
}))

vi.mock('./serviceRegistry', async () => {
  const { create } = await import('zustand')
  const useServiceStore = create(() => ({
    mode: 'fixture' as 'probing' | 'live' | 'fixture',
    health: null,
    sessionId: 'session-under-test',
    services: { search: { search: (request: SearchRequest, signal: AbortSignal) => clients.search(request, signal) } },
    probe: async () => undefined,
  }))
  return { useServiceStore }
})

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason: unknown) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve
    reject = innerReject
  })
  // Aborted requests reject on purpose; nothing may report that as an
  // unhandled rejection in the middle of an otherwise green run.
  promise.catch(() => undefined)
  return { promise, resolve, reject }
}

interface Run {
  ownership: SearchOwnership
  snapshot: Deferred<Record<string, unknown>>
  response: Deferred<SearchResponse>
  request?: SearchRequest
  signal: AbortSignal
}

const runs: Run[] = []

/** A drawing with real content: enough coverage and enough spread. */
const SUBSTANTIVE = { inkPixels: 4096, coverage: 0.0156, bboxDiagonalRatio: 0.62 }
/** A snapshot with nothing visible on it: a cleared canvas, or an all-transparent import. */
const BLANK = { inkPixels: 0, coverage: 0, bboxDiagonalRatio: 0 }

let snapshotInk = SUBSTANTIVE

function lastRun(): Run {
  const run = runs[runs.length - 1]
  if (!run) throw new Error('no search request was issued')
  return run
}

function flush() {
  return act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}

/** Let the debounced request(s) start their snapshots. */
function debounce() {
  return act(async () => {
    vi.advanceTimersByTime(400)
    await Promise.resolve()
    await Promise.resolve()
  })
}

/** Answer one request's snapshot; the hook then decides whether to search. */
function snapshot(run: Run, unavailable: string[] = []) {
  return act(async () => {
    run.snapshot.resolve({
      token: run.ownership.token,
      documentId: run.ownership.documentId,
      revision: run.ownership.revision,
      generation: run.ownership.generation,
      image: new Blob(['snapshot']),
      ink: snapshotInk,
      worker: true,
      unavailable,
    })
    await Promise.resolve()
    await Promise.resolve()
  })
}

/** Answer a request end to end: snapshot, then the gallery's response. */
async function succeed(run: Run) {
  await snapshot(run)
  await act(async () => {
    run.response.resolve({
      revision: run.ownership.revision,
      generation: run.ownership.generation,
      mode: 'confident',
      interpretation: `run ${run.ownership.token}`,
      groups: [],
    })
    await Promise.resolve()
    await Promise.resolve()
  })
}

/** Let a request fail the way a dropped connection does. */
async function fail(run: Run, message: string) {
  await snapshot(run)
  await act(async () => {
    run.response.reject(new Error(message))
    await Promise.resolve()
    await Promise.resolve()
  })
}

/** A new stroke's worth of invalidation: generation bump, ownership dropped. */
function invalidate() {
  return act(async () => {
    useSearchStore.getState().invalidate(false)
    await Promise.resolve()
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  runs.length = 0
  snapshotInk = SUBSTANTIVE
  clients.prepare.mockReset()
  clients.search.mockReset()
  useUiStore.setState({ theme: 'dark' })
  useSearchStore.setState({
    generation: 0,
    drawing: false,
    loading: false,
    error: null,
    response: null,
    textHint: '',
    selectedStyle: null,
    owner: null,
  })
  useDocumentStore.setState({ hasHydrated: true, activeLayerId: 'layer-1' })

  clients.prepare.mockImplementation((_document: DrawingDocument, ownership: SearchOwnership, signal: AbortSignal) => {
    const run: Run = { ownership, snapshot: deferred(), response: deferred(), signal }
    runs.push(run)
    signal.addEventListener(
      'abort',
      () => run.response.reject(new DOMException('Search cancelled', 'AbortError')),
      { once: true },
    )
    return run.snapshot.promise
  })
  clients.search.mockImplementation((request: SearchRequest, signal: AbortSignal) => {
    const run = lastRun()
    run.request = request
    run.signal = signal
    return run.response.promise
  })
})

afterEach(() => {
  vi.useRealTimers()
})

/**
 * Mount the workspace and let it reach a steady state. Startup activates a
 * document (which bumps the revision and the generation), so the claims issued
 * before the debounce are superseded by design and cost no search.
 */
async function mounted() {
  const view = renderHook(() => useWorkspaceLifecycle())
  await flush()
  await debounce()
  expect(runs).toHaveLength(1)
  return view
}

describe('search request ownership', () => {
  it('drops a stale success instead of painting it onto the newer request', async () => {
    await mounted()
    const stale = lastRun()
    expect(useSearchStore.getState().loading).toBe(true)

    await invalidate()
    await debounce()
    const current = lastRun()
    expect(current.ownership.token).not.toBe(stale.ownership.token)
    expect(useSearchStore.getState().loading).toBe(true)

    // The loser answers first. Nothing it writes may land.
    await succeed(stale)
    expect(useSearchStore.getState().response).toBeNull()
    expect(useSearchStore.getState().error).toBeNull()
    // …and the winner is still pending, with its spinner intact.
    expect(useSearchStore.getState().loading).toBe(true)

    await succeed(current)
    expect(useSearchStore.getState().response?.interpretation).toBe(`run ${current.ownership.token}`)
    expect(useSearchStore.getState().loading).toBe(false)
  })

  it('drops a stale failure instead of interrupting the newer request', async () => {
    await mounted()
    const stale = lastRun()
    await act(async () => {
      useSearchStore.getState().setTextHint('a hand')
      await Promise.resolve()
    })
    await debounce()

    await fail(stale, 'the gallery exploded')
    expect(useSearchStore.getState().error).toBeNull()
    expect(useSearchStore.getState().loading).toBe(true)
  })

  it('surfaces the failure of the request that still owns the panel', async () => {
    await mounted()
    await fail(lastRun(), 'the gallery exploded')
    expect(useSearchStore.getState().error).toBe('the gallery exploded')
    expect(useSearchStore.getState().loading).toBe(false)
  })

  it('spends no request on a debounce that was superseded before it started', async () => {
    // Startup itself claims more than once (activating the document bumps the
    // revision, then the generation follows). None of that may cost a snapshot:
    // only the request that survives the debounce is allowed to make one.
    const view = renderHook(() => useWorkspaceLifecycle())
    await flush()
    const claimsBeforeDebounce = useSearchStore.getState().owner?.token ?? 0
    expect(claimsBeforeDebounce).toBeGreaterThan(1)
    expect(runs).toHaveLength(0)

    // Supersede it inside the debounce window as well, then let the surviving
    // request run.
    await invalidate()
    await debounce()
    expect(runs).toHaveLength(1)
    expect(runs[0]?.ownership.token).toBe(useSearchStore.getState().owner?.token)
    view.unmount()
  })

  it('settles loading when a request is aborted after its snapshot resolved', async () => {
    await mounted()
    const run = lastRun()
    await snapshot(run)
    expect(useSearchStore.getState().loading).toBe(true)

    await invalidate()
    expect(useSearchStore.getState().loading).toBe(false)
    expect(run.signal.aborted).toBe(true)
    // The half-finished request can no longer write anything.
    await act(async () => {
      run.response.resolve({
        revision: run.ownership.revision,
        generation: run.ownership.generation,
        mode: 'confident',
        interpretation: 'late',
        groups: [],
      })
      await Promise.resolve()
    })
    expect(useSearchStore.getState().response).toBeNull()
  })

  it('lets the owner settle the spinner but not a superseded request', async () => {
    await mounted()
    const stale = lastRun()
    await invalidate()
    await debounce()
    const current = lastRun()

    // A superseded request's cleanup must not clear the newer request's
    // spinner — that is the "old request finishes, panel stops spinning" bug.
    expect(useSearchStore.getState().release(stale.ownership)).toBe(false)
    expect(useSearchStore.getState().loading).toBe(true)
    expect(useSearchStore.getState().owner?.token).toBe(current.ownership.token)

    // …and the owner's own release does settle it.
    expect(useSearchStore.getState().release(current.ownership)).toBe(true)
    expect(useSearchStore.getState().loading).toBe(false)
    expect(useSearchStore.getState().owner).toBeNull()
  })

  it('unmounting settles the spinner, drops ownership, and voids the answer', async () => {
    const view = await mounted()
    const run = lastRun()
    expect(useSearchStore.getState().loading).toBe(true)

    await act(async () => {
      view.unmount()
    })
    await flush()
    expect(useSearchStore.getState().loading).toBe(false)
    expect(useSearchStore.getState().owner).toBeNull()
    expect(run.signal.aborted).toBe(true)

    await succeed(run)
    expect(useSearchStore.getState().response).toBeNull()
  })
})

describe('search lifecycle identity', () => {
  it('re-issues on a document switch that keeps the revision number', async () => {
    await mounted()
    const previous = lastRun()
    const document = useDocumentStore.getState().document
    const replacement: DrawingDocument = {
      ...structuredClone(document),
      id: 'document-swapped',
      revision: document.revision,
    }

    await act(async () => {
      useDocumentStore.setState({ document: replacement })
      await Promise.resolve()
    })
    await debounce()

    const next = lastRun()
    expect(next.ownership.documentId).toBe('document-swapped')
    expect(next.ownership.revision).toBe(previous.ownership.revision)
    expect(next.ownership.token).not.toBe(previous.ownership.token)
    // The replaced document's answer is worthless now, and it is not adopted.
    await succeed(previous)
    expect(useSearchStore.getState().response).toBeNull()
    expect(useSearchStore.getState().owner?.documentId).toBe('document-swapped')
  })

  it('a service-mode switch settles loading, aborts, and re-issues afterwards', async () => {
    const { useServiceStore } = await import('./serviceRegistry')
    await mounted()
    const before = lastRun()

    await act(async () => {
      useServiceStore.setState({ mode: 'probing' })
      await Promise.resolve()
    })
    expect(useSearchStore.getState().loading).toBe(false)
    expect(before.signal.aborted).toBe(true)

    // Nothing to search with while probing: no queued request either.
    await debounce()
    expect(runs).toHaveLength(1)

    await act(async () => {
      useServiceStore.setState({ mode: 'live' })
      await Promise.resolve()
    })
    await debounce()
    expect(runs).toHaveLength(2)
    expect(useSearchStore.getState().loading).toBe(true)
    await succeed(lastRun())
    expect(useSearchStore.getState().response).not.toBeNull()
    expect(useSearchStore.getState().loading).toBe(false)
  })

  it('restarting the drawing invalidates ownership and the spinner together', async () => {
    await mounted()
    const run = lastRun()
    await act(async () => {
      // Exactly what the canvas does on pointerdown.
      useSearchStore.getState().invalidate(true)
      useSearchStore.getState().setDrawing(true)
      await Promise.resolve()
    })
    expect(useSearchStore.getState().loading).toBe(false)
    expect(useSearchStore.getState().owner).toBeNull()
    // The effect teardown aborted the in-flight request.
    expect(run.signal.aborted).toBe(true)
  })
})

describe('raster sufficiency and the vector branch', () => {
  it('reports counts that match the delivered payload, and searches on ink', async () => {
    await mounted()
    const run = lastRun()
    await snapshot(run)
    expect(clients.search).toHaveBeenCalledOnce()
    const [request] = clients.search.mock.calls[0] as [SearchRequest]
    // An empty vector payload is not sent at all: `stroke_status` has to mean
    // "the stroke branch had input", and a 0-stroke payload under a nonzero
    // stroke_count is exactly the mismatch the API rejects with a 422.
    expect(request.strokes).toBeUndefined()
    expect(request.strokeCount).toBe(0)
    expect(request.pointCount).toBe(0)
    expect(request.rasterCount).toBe(0)
    // Sufficiency travels as a raster measurement, never as a count.
    expect(request.ink).toMatchObject(SUBSTANTIVE)
  })

  it('spends no request on a snapshot with no ink, and says why structurally', async () => {
    snapshotInk = BLANK
    await mounted()
    // The snapshot measured blank, so the request stops there: no gallery call,
    // no spinner, and the reason is a structural degradation rather than a
    // guess the UI would have to re-derive from counts.
    await snapshot(lastRun())
    expect(clients.search).not.toHaveBeenCalled()
    const state = useSearchStore.getState()
    expect(state.loading).toBe(false)
    expect(state.error).toBeNull()
    expect(state.response?.mode).toBe('empty')
    expect(state.response?.groups).toEqual([])
    expect(state.response?.degradations).toContainEqual({
      kind: 'blank_raster',
      detail: 'snapshot carries no ink: nothing was drawn, or the imported image is blank',
    })
  })

  it('searches a substantive raster import that has no vector geometry at all', async () => {
    const document = useDocumentStore.getState().document
    const imported: DrawingDocument = {
      ...structuredClone(document),
      layers: document.layers.map((layer, index) =>
        index === 3
          ? {
              ...layer,
              operations: [
                { id: 'raster-1', kind: 'raster', assetId: 'raster-a', x: 0, y: 0, width: 2048, height: 2048, createdAt: 0 },
              ],
            }
          : layer,
      ),
    }
    await mounted()
    await act(async () => {
      useDocumentStore.setState({ document: imported })
      await Promise.resolve()
    })
    await debounce()
    const run = lastRun()
    await snapshot(run)

    const [request] = clients.search.mock.calls.at(-1) as [SearchRequest]
    // The import is content, and it is content the vector counts cannot
    // describe — so it is reported separately instead of inflating the stroke
    // counts the payload is checked against.
    expect(request.rasterCount).toBe(1)
    expect(request.strokeCount).toBe(0)
    expect(request.strokes).toBeUndefined()
    expect(request.ink).toMatchObject(SUBSTANTIVE)
    await succeed(run)
    expect(useSearchStore.getState().response).not.toBeNull()
  })

  it('never silences search because the vector counts are thin', async () => {
    const document = useDocumentStore.getState().document
    const early: DrawingDocument = {
      ...structuredClone(document),
      layers: document.layers.map((layer, index) =>
        index === 0
          ? {
              ...layer,
              operations: [
                {
                  id: 'stroke-1',
                  kind: 'stroke',
                  tool: 'monoline',
                  size: 12,
                  smoothing: 0.1,
                  streamline: 0,
                  simulatePressure: false,
                  createdAt: 0,
                  points: [
                    { x: 10, y: 10, pressure: 0.5, time: 0 },
                    { x: 20, y: 30, pressure: 0.5, time: 16 },
                  ],
                },
              ],
            }
          : layer,
      ),
    }
    await mounted()
    await act(async () => {
      useDocumentStore.setState({ document: early })
      await Promise.resolve()
    })
    await debounce()
    const run = lastRun()
    await snapshot(run)

    // Two sampled points: far below the 20 the stroke branch wants, but the
    // raster has ink, so the query is issued and the payload is delivered.
    const [request] = clients.search.mock.calls.at(-1) as [SearchRequest]
    expect(request.strokeCount).toBe(1)
    expect(request.pointCount).toBe(2)
    expect(request.strokes?.strokes).toHaveLength(1)
    await succeed(run)
    expect(useSearchStore.getState().response?.mode).toBe('confident')
  })

  it('stops before searching when the snapshot no longer answers its request', async () => {
    await mounted()
    const run = lastRun()
    await invalidate()
    // A snapshot echoed for the superseded identity: even if it arrives after
    // the new claim, the identity check drops it before a request is made.
    await act(async () => {
      run.snapshot.resolve({
        token: run.ownership.token,
        documentId: run.ownership.documentId,
        revision: run.ownership.revision,
        // The generation the *new* request is running under: a mismatch.
        generation: run.ownership.generation + 1,
        image: new Blob(['snapshot']),
        ink: snapshotInk,
        worker: true,
        unavailable: [],
      })
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(clients.search).not.toHaveBeenCalled()
  })
})

describe('a stored import blob that is gone', () => {
  /** The hook composes the warning; the gallery's own note must survive the append. */
  async function succeedWithWarning(run: Run, warning: string | null) {
    await act(async () => {
      run.response.resolve({
        revision: run.ownership.revision,
        generation: run.ownership.generation,
        mode: 'confident',
        interpretation: `run ${run.ownership.token}`,
        groups: [],
        warning,
      })
      await Promise.resolve()
      await Promise.resolve()
    })
  }

  it('searches the surviving ink and says what was left out on the warning channel', async () => {
    await mounted()
    const run = lastRun()
    await snapshot(run, ['raster-gone'])
    // A degraded input, never a veto: the request still goes out.
    expect(clients.search).toHaveBeenCalledOnce()
    await succeedWithWarning(run, 'gallery-side note')
    const warning = useSearchStore.getState().response?.warning
    expect(warning).toContain('gallery-side note')
    expect(warning).toContain('One imported artwork is no longer stored on this device')
    // The omission is not a degradations kind: that list is the gallery's
    // attestation about a query it ranked, and it knows nothing about this
    // device's storage.
    expect(useSearchStore.getState().response?.degradations ?? []).not.toContainEqual(
      expect.objectContaining({ detail: expect.stringContaining('no longer stored') }),
    )
  })

  it('keeps the omission visible next to a blank-canvas verdict that never reaches the gallery', async () => {
    snapshotInk = BLANK
    await mounted()
    await snapshot(lastRun(), ['raster-gone'])
    expect(clients.search).not.toHaveBeenCalled()
    const response = useSearchStore.getState().response
    expect(response?.mode).toBe('empty')
    expect(response?.warning).toContain('snapshot carries no ink')
    expect(response?.warning).toContain('One imported artwork is no longer stored on this device')
  })

  it('mentions nothing when every referenced import resolved', async () => {
    await mounted()
    const run = lastRun()
    await succeed(run)
    expect(useSearchStore.getState().response?.warning ?? null).toBeNull()
  })
})
