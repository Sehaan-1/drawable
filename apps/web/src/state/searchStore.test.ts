import { beforeEach, describe, expect, it, vi } from 'vitest'
import { PINS_VERSION, useSearchStore } from './searchStore'
import {
  LEGACY_FIXTURE_PINS_KEY,
  pinsKey,
  readLocalPins,
  writeLocalPins,
} from '../services/pinStorage'
import { localPinClient, type PinClient, type PinSnapshot } from '../services/frontendServices'
import type { ReferenceAsset } from '../lib/types'

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
