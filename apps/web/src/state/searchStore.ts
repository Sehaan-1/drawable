import { create } from 'zustand'
import type { ReferenceAsset, SearchResponse } from '../lib/types'

/**
 * Pinned references are durable *local application state*, fully independent
 * of learning: the API's preference profile never sees them. The fixture
 * gallery and the live gallery use separate namespaces so a pin made against
 * synthetic fixtures can never leak into a real-gallery session, and the
 * `:version` suffix lets the storage format evolve without a risky rewrite.
 */

export type GalleryKind = 'fixture' | 'live'

export const PINS_VERSION = 1
const LEGACY_PINS_KEY = 'drawable-fixture-pins'

export function pinsKey(kind: GalleryKind): string {
  return `drawable-pins:${kind}:${PINS_VERSION}`
}

function readPins(kind: GalleryKind): ReferenceAsset[] {
  const key = pinsKey(kind)
  if (localStorage.getItem(key) === null && kind === 'fixture') {
    // One-time migration: the pre-v2 app stored fixture pins under a single
    // un-namespaced key, so they can only ever belong to the fixture
    // namespace — never to whichever gallery is read first.
    const legacy = localStorage.getItem(LEGACY_PINS_KEY)
    if (legacy !== null) {
      localStorage.setItem(key, legacy)
      localStorage.removeItem(LEGACY_PINS_KEY)
    }
  }
  try {
    return JSON.parse(localStorage.getItem(key) ?? '[]') as ReferenceAsset[]
  } catch {
    return []
  }
}

function writePins(kind: GalleryKind, pinned: ReferenceAsset[]): void {
  localStorage.setItem(pinsKey(kind), JSON.stringify(pinned))
}

interface SearchState {
  /** Which gallery namespace pins are read from / written to. */
  gallery: GalleryKind
  generation: number
  drawing: boolean
  loading: boolean
  error: string | null
  response: SearchResponse | null
  textHint: string
  selectedStyle: string | null
  selectedAsset: ReferenceAsset | null
  pinned: ReferenceAsset[]
  /** Switch the pin namespace and reload pins for that gallery. */
  setGallery: (kind: GalleryKind) => void
  invalidate: (drawing?: boolean) => number
  setDrawing: (drawing: boolean) => void
  setLoading: (loading: boolean) => void
  setResponse: (response: SearchResponse) => void
  setError: (error: string | null) => void
  setTextHint: (hint: string) => void
  setSelectedStyle: (style: string | null) => void
  setSelectedAsset: (asset: ReferenceAsset | null) => void
  togglePin: (asset: ReferenceAsset) => void
}

export const useSearchStore = create<SearchState>((set, get) => ({
  gallery: 'fixture',
  generation: 0,
  drawing: false,
  loading: false,
  error: null,
  response: null,
  textHint: '',
  selectedStyle: null,
  selectedAsset: null,
  pinned: readPins('fixture'),
  setGallery: (kind) => {
    if (get().gallery === kind) return
    set({ gallery: kind, pinned: readPins(kind) })
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
  togglePin: (asset) => set((state) => {
    const exists = state.pinned.some((item) => item.id === asset.id)
    const pinned = exists ? state.pinned.filter((item) => item.id !== asset.id) : [asset, ...state.pinned]
    writePins(state.gallery, pinned)
    return { pinned }
  }),
}))
