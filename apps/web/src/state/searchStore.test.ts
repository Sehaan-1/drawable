import { beforeEach, describe, expect, it, vi } from 'vitest'
import { PINS_VERSION, useSearchStore } from './searchStore'
import {
  LEGACY_FIXTURE_PINS_KEY,
  pinsKey,
  readLocalPins,
  writeLocalPins,
} from '../services/pinStorage'
import { localPinClient, type PinClient, type PinSnapshot } from '../services/frontendServices'
import type { ReferenceAsset, SearchResponse } from '../lib/types'

/**
 * Pins are durable state, namespaced per gallery. The fixture namespace is an
 * isolated local store; the live namespace is owned by the API. Neither may
 * ever read the other's assets, and a reload/restart must bring them back.
 */

function asset(id: string, over: Partial<ReferenceAsset> = {}): ReferenceAsset {
  return {
    id,
    title: id,
    imageUrl: `data:image/svg+xml,${id}`,
    style: 'Cartoon',
    scope: 'Eye',
    source: 'Test',
    native: true,
    match: 'Strong',
    traceAllowed: true,
    traceUrl: `data:image/svg+xml,${id}`,
    ...over,
  }
}

/** A stand-in for the API-backed live pin store. */
function fakeLiveClient(initial: ReferenceAsset[] = []): PinClient & { pins: ReferenceAsset[] } {
  const state = { pins: [...initial] }
  const snapshot = (): PinSnapshot => ({ pins: [...state.pins], revoked: [] })
  return {
    get pins() {
      return state.pins
    },
    set pins(next: ReferenceAsset[]) {
      state.pins = next
    },
    list: async () => snapshot(),
    pin: async (item) => {
      state.pins = [item, ...state.pins.filter((pin) => pin.id !== item.id)]
      return snapshot()
    },
    unpin: async (assetId) => {
      state.pins = state.pins.filter((pin) => pin.id !== assetId)
      return snapshot()
    },
  }
}

beforeEach(() => {
  localStorage.clear()
  useSearchStore.setState({
    gallery: 'fixture',
    pinClient: localPinClient(),
    pinned: [],
    revokedPins: [],
    pinsHydrated: false,
    pinError: null,
    generation: 0,
    drawing: false,
    loading: false,
    error: null,
    response: null,
    owner: null,
    textHint: '',
    selectedStyle: null,
  })
})

describe('pins namespace keys', () => {
  it('derives namespaced keys with a version suffix', () => {
    expect(pinsKey('fixture')).toBe(`drawable-pins:fixture:${PINS_VERSION}`)
    expect(pinsKey('live')).toBe(`drawable-pins:live:${PINS_VERSION}`)
  })
})

describe('attachGallery', () => {
  it('hydrates the fixture namespace from its durable local store', async () => {
    writeLocalPins('fixture', [asset('fixture-1')])
    await useSearchStore.getState().attachGallery('fixture', localPinClient())
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['fixture-1'])
  })

  it('swaps the whole pin set when the gallery changes and never mixes the two', async () => {
    writeLocalPins('fixture', [asset('fixture-1')])
    const live = fakeLiveClient([asset('ls_synthetic_live0000000000001')])

    await useSearchStore.getState().attachGallery('fixture', localPinClient())
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['fixture-1'])

    await useSearchStore.getState().attachGallery('live', live)
    expect(useSearchStore.getState().gallery).toBe('live')
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual([
      'ls_synthetic_live0000000000001',
    ])

    await useSearchStore.getState().attachGallery('fixture', localPinClient())
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['fixture-1'])
    // Live pins were never written to local storage.
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()
  })

  it('surfaces revoked pins reported by revalidation', async () => {
    const client: PinClient = {
      list: async () => ({ pins: [], revoked: [{ assetId: 'ls_x', reasons: ['display_not_permitted'] }] }),
      pin: async () => ({ pins: [], revoked: [] }),
      unpin: async () => ({ pins: [], revoked: [] }),
    }
    await useSearchStore.getState().attachGallery('live', client)
    expect(useSearchStore.getState().pinned).toEqual([])
    expect(useSearchStore.getState().revokedPins).toEqual([
      { assetId: 'ls_x', reasons: ['display_not_permitted'] },
    ])
    useSearchStore.getState().dismissRevokedPins()
    expect(useSearchStore.getState().revokedPins).toEqual([])
  })

  it('keeps working when the durable store is unreachable', async () => {
    const client: PinClient = {
      list: async () => {
        throw new Error('API unreachable')
      },
      pin: async () => ({ pins: [], revoked: [] }),
      unpin: async () => ({ pins: [], revoked: [] }),
    }
    await useSearchStore.getState().attachGallery('live', client)
    expect(useSearchStore.getState().pinError).toBe('API unreachable')
    expect(useSearchStore.getState().pinned).toEqual([])
  })
})

describe('togglePin', () => {
  it('writes to the active gallery store only, and survives a reload', async () => {
    await useSearchStore.getState().attachGallery('fixture', localPinClient())
    expect(await useSearchStore.getState().togglePin(asset('a'))).toBe(true)

    expect(readLocalPins('fixture').map((item) => item.id)).toEqual(['a'])
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()

    // A fresh store instance (as after a reload) hydrates the same pin.
    useSearchStore.setState({ pinned: [], pinsHydrated: false })
    await useSearchStore.getState().hydratePins()
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['a'])
  })

  it('toggling the same asset twice removes it again', async () => {
    await useSearchStore.getState().attachGallery('fixture', localPinClient())
    expect(await useSearchStore.getState().togglePin(asset('a'))).toBe(true)
    expect(await useSearchStore.getState().togglePin(asset('a'))).toBe(false)
    expect(useSearchStore.getState().pinned).toEqual([])
    expect(readLocalPins('fixture')).toEqual([])
  })

  it('delegates live pins to the injected client and never to local storage', async () => {
    const live = fakeLiveClient()
    const spy = vi.spyOn(live, 'pin')
    await useSearchStore.getState().attachGallery('live', live)
    await useSearchStore.getState().togglePin(asset('ls_synthetic_aaaaaaaaaaaaaaaa'))
    expect(spy).toHaveBeenCalledOnce()
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual([
      'ls_synthetic_aaaaaaaaaaaaaaaa',
    ])
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()
    expect(localStorage.getItem(pinsKey('fixture'))).toBeNull()
  })

  it('reports the unchanged state when the durable write fails', async () => {
    const client: PinClient = {
      list: async () => ({ pins: [], revoked: [] }),
      pin: async () => {
        throw new Error('nope')
      },
      unpin: async () => ({ pins: [], revoked: [] }),
    }
    await useSearchStore.getState().attachGallery('live', client)
    expect(await useSearchStore.getState().togglePin(asset('ls_x'))).toBe(false)
    expect(useSearchStore.getState().pinned).toEqual([])
    expect(useSearchStore.getState().pinError).toBe('nope')
  })
})

describe('legacy migration', () => {
  it('migrates the un-namespaced key into the fixture namespace once', async () => {
    const legacy = [asset('legacy-1'), asset('legacy-2')]
    localStorage.setItem(LEGACY_FIXTURE_PINS_KEY, JSON.stringify(legacy))

    // Reading the live namespace must never absorb them.
    await useSearchStore.getState().attachGallery('live', fakeLiveClient())
    expect(useSearchStore.getState().pinned).toEqual([])
    expect(localStorage.getItem(LEGACY_FIXTURE_PINS_KEY)).not.toBeNull()

    await useSearchStore.getState().attachGallery('fixture', localPinClient())
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['legacy-1', 'legacy-2'])
    expect(localStorage.getItem(LEGACY_FIXTURE_PINS_KEY)).toBeNull()
    expect(readLocalPins('fixture').map((item) => item.id)).toEqual(['legacy-1', 'legacy-2'])
  })
})

describe('search request ownership', () => {
  const identity = { documentId: 'doc-a', revision: 1, generation: 3 }
  const forRevision = (revision: number, generation: number) => ({
    ...identity,
    revision,
    generation,
  })
  const answer = (over: Partial<SearchResponse> = {}): SearchResponse =>
    ({
      revision: identity.revision,
      generation: identity.generation,
      mode: 'confident',
      interpretation: 'Face construction',
      groups: [],
      ...over,
    }) as SearchResponse

  it('gives ownership to the newest claim and settles the superseded request', () => {
    const stale = useSearchStore.getState().claim(identity)
    expect(useSearchStore.getState().begin(stale)).toBe(true)
    expect(useSearchStore.getState().loading).toBe(true)

    // A different document claims the same generation number: the old request
    // loses the right to write, and its spinner cannot strand the new one.
    const fresh = useSearchStore.getState().claim(forRevision(2, identity.generation))
    const state = useSearchStore.getState()
    expect(state.owner?.token).toBe(fresh.token)
    expect(state.loading).toBe(false)
    expect(state.begin(stale)).toBe(false)
    expect(state.resolve(stale, answer())).toBe(false)
    expect(state.reject(stale, 'boom')).toBe(false)
    expect(state.release(stale)).toBe(false)
    expect(state.response).toBeNull()
    expect(state.error).toBeNull()

    expect(useSearchStore.getState().resolve(fresh, answer({ revision: 2 }))).toBe(true)
    expect(useSearchStore.getState().response?.mode).toBe('confident')
    expect(useSearchStore.getState().loading).toBe(false)
    expect(useSearchStore.getState().owner).toBeNull()
  })

  it('distinguishes two claims of an identical drawing state by token', () => {
    // A retry after a failure has the same document, revision, and generation as
    // the attempt it replaces — the token is what keeps them apart.
    const first = useSearchStore.getState().claim(identity)
    const second = useSearchStore.getState().claim(identity)
    expect(second.token).not.toBe(first.token)
    expect(useSearchStore.getState().isOwner(first)).toBe(false)
    expect(useSearchStore.getState().isOwner(second)).toBe(true)
    expect(useSearchStore.getState().resolve(first, answer())).toBe(false)
    expect(useSearchStore.getState().resolve(second, answer())).toBe(true)
  })

  it('drops an answer whose echoed identity is not the request it owns', () => {
    const owner = useSearchStore.getState().claim(identity)
    useSearchStore.getState().begin(owner)
    // The gallery replied for a different revision: it is not applied, and it
    // does not clear the spinner either — only a real transition may.
    expect(useSearchStore.getState().resolve(owner, answer({ revision: 99 }))).toBe(false)
    expect(useSearchStore.getState().response).toBeNull()
    expect(useSearchStore.getState().loading).toBe(true)
    expect(useSearchStore.getState().owner?.token).toBe(owner.token)
    expect(useSearchStore.getState().resolve(owner, answer({ generation: 77 }))).toBe(false)
    expect(useSearchStore.getState().resolve(owner, answer())).toBe(true)
  })

  it('releases only for the owner, so old cleanup cannot unset a newer spinner', () => {
    const stale = useSearchStore.getState().claim(identity)
    const current = useSearchStore.getState().claim(forRevision(2, 4))
    useSearchStore.getState().begin(current)
    expect(useSearchStore.getState().release(stale)).toBe(false)
    expect(useSearchStore.getState().loading).toBe(true)
    expect(useSearchStore.getState().owner?.token).toBe(current.token)
    expect(useSearchStore.getState().release(current)).toBe(true)
    expect(useSearchStore.getState().loading).toBe(false)
    expect(useSearchStore.getState().owner).toBeNull()
  })

  it('settles ownership and the spinner together on every invalidating transition', () => {
    const owner = useSearchStore.getState().claim(identity)
    useSearchStore.getState().begin(owner)
    const before = useSearchStore.getState().generation
    expect(useSearchStore.getState().loading).toBe(true)

    expect(useSearchStore.getState().invalidate(true)).toBe(before + 1)
    expect(useSearchStore.getState().owner).toBeNull()
    expect(useSearchStore.getState().loading).toBe(false)
    expect(useSearchStore.getState().drawing).toBe(true)
    expect(useSearchStore.getState().generation).toBe(before + 1)

    // A query change takes the same route, in a single write, and truncates the
    // hint — "superseded but still spinning" must never be observable.
    const afterHint = useSearchStore.getState().claim(forRevision(1, before + 1))
    useSearchStore.getState().begin(afterHint)
    useSearchStore.getState().setTextHint('x'.repeat(200))
    expect(useSearchStore.getState().generation).toBe(before + 2)
    expect(useSearchStore.getState().textHint).toHaveLength(120)
    expect(useSearchStore.getState().owner).toBeNull()
    expect(useSearchStore.getState().loading).toBe(false)

    const afterStyle = useSearchStore.getState().claim(forRevision(1, before + 2))
    useSearchStore.getState().begin(afterStyle)
    useSearchStore.getState().setSelectedStyle('Manga / anime')
    expect(useSearchStore.getState().generation).toBe(before + 3)
    expect(useSearchStore.getState().selectedStyle).toBe('Manga / anime')
    expect(useSearchStore.getState().owner).toBeNull()
    expect(useSearchStore.getState().loading).toBe(false)
  })

  it('writes the response and the error through the owner only', () => {
    const owner = useSearchStore.getState().claim(identity)
    useSearchStore.getState().resolve(owner, answer({ warning: 'degraded' }))
    expect(useSearchStore.getState().response?.warning).toBe('degraded')
    expect(useSearchStore.getState().error).toBeNull()

    const retry = useSearchStore.getState().claim(forRevision(2, identity.generation))
    expect(useSearchStore.getState().begin(retry)).toBe(true)
    expect(useSearchStore.getState().reject(retry, 'Search failed')).toBe(true)
    const state = useSearchStore.getState()
    expect(state.error).toBe('Search failed')
    expect(state.loading).toBe(false)
    // The last good answer is kept as provenance (`recordInteraction` reports
    // its revision) and stays hidden: the dock renders the error ahead of any
    // response, so a superseded result cannot be mistaken for the current one.
    expect(state.response?.revision).toBe(1)
    expect(useSearchStore.getState().reject(retry, 'again')).toBe(false)
  })
})
