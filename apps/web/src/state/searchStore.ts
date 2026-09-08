import { create } from 'zustand'
import type { GalleryKind } from '@drawable/contracts'
import type { ReferenceAsset, SearchResponse } from '../lib/types'
import { localPinClient, type PinClient, type RevokedPinView } from '../services/frontendServices'

/**
 * Pinned references are durable *state*, fully independent of learning: an
 * affinity reset never clears them, and turning learning off never stops
 * them from working.
 *
 * Where they are stored depends on the gallery, and the two are isolated:
 *
 * * **live** pins belong to the API's SQLite database, so they survive a page
 *   reload *and* a restart of the API, and every read revalidates them
 *   against the current gallery (a pin whose asset lost permission or
 *   eligibility comes back as `revokedPins`, not as a pin);
 * * **fixture** pins are offline-only and live in an isolated local store
 *   (see `services/pinStorage.ts`).
 *
 * The store never talks to a gallery directly; `attachGallery` injects the
 * client for the active one, which is also what keeps the two namespaces from
 * ever being read into each other.
 *
 * Search state is the opposite of durable state: it belongs to exactly one
 * in-flight request, identified by a `SearchOwnership` token tied to the
 * document id, the document revision, *and* the search generation. Success,
 * failure, and both directions of the loading spinner are gated on that token
 * (`begin`/`resolve`/`reject`/`release`), because a loser of the race that
 * compares only "is the generation still mine?" can otherwise clear a newer
 * request's spinner or overwrite its results.
 */

export type { GalleryKind }

export { PINS_VERSION, pinsKey } from '../services/pinStorage'

/** The drawing state one search request was issued for. */
export interface SearchIdentity {
  documentId: string
  revision: number
  generation: number
}

/**
 * Identity plus the monotonic claim token that authorizes writes for it.
 *
 * `token` distinguishes two claims over the same document/revision/generation
 * (a retry after a failure, or a re-run caused by the service registry
 * settling), which no subset of the identity fields can do.
 */
export interface SearchOwnership extends SearchIdentity {
  token: number
}

interface SearchState {
  /** Which gallery namespace pins are read from / written to. */
  gallery: GalleryKind
  pinClient: PinClient
  generation: number
  drawing: boolean
  loading: boolean
  error: string | null
  response: SearchResponse | null
  textHint: string
  selectedStyle: string | null
  selectedAsset: ReferenceAsset | null
  pinned: ReferenceAsset[]
  /** Pins dropped by revalidation since the last hydrate. */
  revokedPins: RevokedPinView[]
  pinsHydrated: boolean
  pinError: string | null
  /** The request currently allowed to write results, errors, and the spinner. */
  owner: SearchOwnership | null
  /** Point the store at a gallery's pin store and hydrate from it. */
  attachGallery: (kind: GalleryKind, client: PinClient) => Promise<void>
  /** Re-read pins from their durable store (startup, or after a change). */
  hydratePins: () => Promise<void>
  invalidate: (drawing?: boolean) => number
  setDrawing: (drawing: boolean) => void
  setTextHint: (hint: string) => void
  setSelectedStyle: (style: string | null) => void
  setSelectedAsset: (asset: ReferenceAsset | null) => void
  togglePin: (asset: ReferenceAsset) => Promise<boolean>
  dismissRevokedPins: () => void
  /**
   * Take ownership of the search lifecycle for one drawing identity.
   *
   * Claiming supersedes whatever request held ownership before — that request
   * can no longer write anything — and settles the spinner, so an interrupted
   * request never leaves "loading" attached to a document it never searched.
   */
  claim: (identity: SearchIdentity) => SearchOwnership
  /** True while `candidate` still matches the whole owning identity. */
  isOwner: (candidate: SearchOwnership) => boolean
  /** Raise the spinner, but only for the current owner. */
  begin: (candidate: SearchOwnership) => boolean
  /** Deliver a response; only the owner may, and only its own identity. */
  resolve: (candidate: SearchOwnership, response: SearchResponse) => boolean
  /** Surface a failure; only the owner may, and only its own identity. */
  reject: (candidate: SearchOwnership, error: string) => boolean
  /**
   * Give up ownership and settle the spinner — but only if `candidate` is
   * *still* the owner. A superseded request's cleanup must not clear a newer
   * request's loading state, so releasing a stale token changes nothing.
   */
  release: (candidate: SearchOwnership) => boolean
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Pinned references are unavailable.'
}

/**
 * Full-identity comparison: the token *and* the document/revision/generation it
 * was claimed for. Matching on a subset is exactly how a stale request used to
 * be able to write another request's state.
 */
function owned(current: SearchOwnership | null, candidate: SearchOwnership): boolean {
  return (
    current !== null &&
    current.token === candidate.token &&
    current.documentId === candidate.documentId &&
    current.revision === candidate.revision &&
    current.generation === candidate.generation
  )
}

export const useSearchStore = create<SearchState>((set, get) => {
  let issued = 0
  return {
    gallery: 'fixture',
    pinClient: localPinClient(),
    generation: 0,
    drawing: false,
    loading: false,
    error: null,
    response: null,
    textHint: '',
    selectedStyle: null,
    selectedAsset: null,
    pinned: [],
    revokedPins: [],
    pinsHydrated: false,
    pinError: null,
    owner: null,
    attachGallery: async (kind, client) => {
      if (get().gallery === kind && get().pinClient === client && get().pinsHydrated) return
      // Drop the previous gallery's pins immediately: showing them against
      // another gallery, even for one frame, would mix the two namespaces.
      set({ gallery: kind, pinClient: client, pinned: [], revokedPins: [], pinsHydrated: false })
      await get().hydratePins()
    },
    hydratePins: async () => {
      const client = get().pinClient
      try {
        const snapshot = await client.list()
        if (get().pinClient !== client) return
        set({ pinned: snapshot.pins, revokedPins: snapshot.revoked, pinsHydrated: true, pinError: null })
      } catch (error) {
        if (get().pinClient !== client) return
        set({ pinError: message(error), pinsHydrated: true })
      }
    },
    invalidate: (drawing = get().drawing) => {
      const generation = get().generation + 1
      set({ generation, drawing, owner: null, loading: false })
      return generation
    },
    setDrawing: (drawing) => set({ drawing }),
    // A new query supersedes the in-flight one: the generation bumps, the
    // request loses ownership, and its spinner is settled in the same write, so
    // no render can observe "superseded but still loading".
    setTextHint: (textHint) =>
      set((state) => ({ textHint: textHint.slice(0, 120), generation: state.generation + 1, owner: null, loading: false })),
    setSelectedStyle: (selectedStyle) =>
      set((state) => ({ selectedStyle, generation: state.generation + 1, owner: null, loading: false })),
    setSelectedAsset: (selectedAsset) => set({ selectedAsset }),
    claim: (identity) => {
      issued += 1
      const owner: SearchOwnership = { token: issued, ...identity }
      // Ownership transfers without touching the previous request's spinner:
      // the *claiming* effect's cleanup is what settles it, and the new request
      // raises it again when it actually starts (see `begin`).
      set({ owner, loading: false })
      return owner
    },
    isOwner: (candidate) => owned(get().owner, candidate),
    begin: (candidate) => {
      if (!owned(get().owner, candidate)) return false
      set({ loading: true })
      return true
    },
    resolve: (candidate, response) => {
      if (!owned(get().owner, candidate)) return false
      // The gallery echoes the identity it was queried with; a mismatch is a
      // late answer for a different drawing and is dropped, not applied.
      if (response.revision !== candidate.revision || response.generation !== candidate.generation) return false
      set({ response, error: null, loading: false, owner: null })
      return true
    },
    reject: (candidate, error) => {
      if (!owned(get().owner, candidate)) return false
      set({ error, loading: false, owner: null })
      return true
    },
    release: (candidate) => {
      if (!owned(get().owner, candidate)) return false
      set({ owner: null, loading: false })
      return true
    },
    /**
     * Pin or unpin, and adopt the durable store's answer. Returns the resulting
     * pinned state so callers can log the matching interaction — the learning
     * signal follows the state change, it never replaces it.
     */
    togglePin: async (asset) => {
      const client = get().pinClient
      const pinned = get().pinned.some((item) => item.id === asset.id)
      try {
        const snapshot = pinned ? await client.unpin(asset.id) : await client.pin(asset)
        if (get().pinClient !== client) return !pinned
        set({ pinned: snapshot.pins, revokedPins: snapshot.revoked, pinError: null })
        return !pinned
      } catch (error) {
        if (get().pinClient === client) set({ pinError: message(error) })
        return pinned
      }
    },
    dismissRevokedPins: () => set({ revokedPins: [] }),
  }
})
