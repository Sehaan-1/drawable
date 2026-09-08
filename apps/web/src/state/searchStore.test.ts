import { beforeEach, describe, expect, it } from 'vitest'
import { PINS_VERSION, pinsKey, useSearchStore, type GalleryKind } from './searchStore'
import type { ReferenceAsset } from '../lib/types'

/**
 * Pins are durable local application state, namespaced per gallery kind so a
 * fixture pin can never appear in a live-gallery session (and vice versa).
 */

function pin(id: string): ReferenceAsset {
  return {
    id,
    title: id,
    imageUrl: `/thumb/${id}.png`,
    style: 'Cartoon',
    scope: 'Eye',
    source: 'Test',
    native: true,
    match: 'Strong',
    traceAllowed: true,
  }
}

beforeEach(() => {
  localStorage.clear()
  useSearchStore.setState({ gallery: 'fixture', pinned: [] })
})

describe('pins namespace keys', () => {
  it('derives namespaced keys with a version suffix', () => {
    expect(pinsKey('fixture')).toBe(`drawable-pins:fixture:${PINS_VERSION}`)
    expect(pinsKey('live')).toBe(`drawable-pins:live:${PINS_VERSION}`)
  })
})

describe('setGallery', () => {
  it('reloads pins from the new namespace when the gallery changes', () => {
    const fixturePin = pin('fixture-1')
    const livePin = pin('live-1')
    localStorage.setItem(pinsKey('fixture'), JSON.stringify([fixturePin]))
    localStorage.setItem(pinsKey('live'), JSON.stringify([livePin]))

    useSearchStore.getState().setGallery('live' as GalleryKind)
    expect(useSearchStore.getState().gallery).toBe('live')
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['live-1'])

    useSearchStore.getState().setGallery('fixture')
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['fixture-1'])
  })

  it('is a no-op when the gallery kind does not change', () => {
    localStorage.setItem(pinsKey('fixture'), JSON.stringify([pin('fixture-1')]))
    useSearchStore.getState().setGallery('fixture')
    // Still empty: the store did not re-read storage for the same kind.
    expect(useSearchStore.getState().pinned).toEqual([])
  })

  it('migrates the legacy un-namespaced key into the fixture namespace once', () => {
    const legacy = [pin('legacy-1'), pin('legacy-2')]
    localStorage.setItem('drawable-fixture-pins', JSON.stringify(legacy))

    useSearchStore.getState().setGallery('live')
    // The legacy key held *fixture* pins; reading the live namespace must
    // never absorb them.
    expect(useSearchStore.getState().pinned).toEqual([])
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()

    useSearchStore.getState().setGallery('fixture')
    expect(useSearchStore.getState().pinned.map((item) => item.id)).toEqual(['legacy-1', 'legacy-2'])
    expect(localStorage.getItem('drawable-pins:fixture:1')).toBe(JSON.stringify(legacy))
    expect(localStorage.getItem('drawable-fixture-pins')).toBeNull()
  })
})

describe('togglePin', () => {
  it('writes to the active gallery namespace only', () => {
    useSearchStore.getState().setGallery('fixture')
    useSearchStore.getState().togglePin(pin('a'))

    expect(JSON.parse(localStorage.getItem(pinsKey('fixture')) ?? '[]')).toHaveLength(1)
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()

    useSearchStore.getState().setGallery('live')
    expect(useSearchStore.getState().pinned).toEqual([])
    useSearchStore.getState().togglePin(pin('b'))
    expect(JSON.parse(localStorage.getItem(pinsKey('live')) ?? '[]').map((item: ReferenceAsset) => item.id)).toEqual(['b'])
    expect(JSON.parse(localStorage.getItem(pinsKey('fixture')) ?? '[]')).toHaveLength(1)
  })

  it('toggling the same asset twice removes it again', () => {
    useSearchStore.getState().togglePin(pin('a'))
    useSearchStore.getState().togglePin(pin('a'))
    expect(useSearchStore.getState().pinned).toEqual([])
    expect(JSON.parse(localStorage.getItem(pinsKey('fixture')) ?? '[]')).toEqual([])
  })
})
