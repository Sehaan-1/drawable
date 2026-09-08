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
 */

export type { GalleryKind }

export { PINS_VERSION, pinsKey } from '../services/pinStorage'

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
  /** Point the store at a gallery's pin store and hydrate from it. */
  attachGallery: (kind: GalleryKind, client: PinClient) => Promise<void>
  /** Re-read pins from their durable store (startup, or after a change). */
  hydratePins: () => Promise<void>
  invalidate: (drawing?: boolean) => number
  setDrawing: (drawing: boolean) => void
  setLoading: (loading: boolean) => void
  setResponse: (response: SearchResponse) => void
  setError: (error: string | null) => void
  setTextHint: (hint: string) => void
  setSelectedStyle: (style: string | null) => void
  setSelectedAsset: (asset: ReferenceAsset | null) => void
  togglePin: (asset: ReferenceAsset) => Promise<boolean>
  dismissRevokedPins: () => void
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Pinned references are unavailable.'
}

export const useSearchStore = create<SearchState>((set, get) => ({
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
    set({ generation, drawing })
    return generation
  },
  setDrawing: (drawing) => set({ drawing }),
  setLoading: (loading) => set({ loading }),
  setResponse: (response) => set({ response, error: null, loading: false }),
  setError: (error) => set({ error, loading: false }),
  setTextHint: (textHint) => set((state) => ({ textHint: textHint.slice(0, 120), generation: state.generation + 1 })),
  setSelectedStyle: (selectedStyle) => set((state) => ({ selectedStyle, generation: state.generation + 1 })),
  setSelectedAsset: (selectedAsset) => set({ selectedAsset }),
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
}))
