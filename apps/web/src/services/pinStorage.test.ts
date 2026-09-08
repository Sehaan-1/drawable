import { beforeEach, describe, expect, it } from 'vitest'
import {
  LEGACY_FIXTURE_PINS_KEY,
  MAX_LOCAL_PINS,
  isLiveAssetId,
  migrateLegacyFixturePins,
  pinsKey,
  readLocalPins,
  sanitizePin,
  takeLegacyLivePinIds,
  writeLocalPins,
} from './pinStorage'
import type { ReferenceAsset } from '../lib/types'

/**
 * Everything in the local pin store is untrusted input. It is validated on
 * read, permission-bearing fields fail closed, and the fixture namespace can
 * never adopt a live gallery asset (or the other way round).
 */

const valid = {
  id: 'fixture-1',
  title: 'Eye study 01',
  imageUrl: 'data:image/svg+xml,%3Csvg%3E',
  style: 'Cartoon',
  scope: 'Eye',
  source: 'drawable procedural fixture',
  native: true,
  match: 'Strong',
  traceAllowed: true,
}

beforeEach(() => localStorage.clear())

describe('sanitizePin', () => {
  it('accepts a well-formed fixture pin and derives its trace source', () => {
    const pin = sanitizePin(valid, 'fixture')
    expect(pin?.id).toBe('fixture-1')
    expect(pin?.traceAllowed).toBe(true)
    expect(pin?.traceUrl).toBe(valid.imageUrl)
  })

  it('fails closed on trace permission', () => {
    expect(sanitizePin({ ...valid, traceAllowed: undefined }, 'fixture')?.traceAllowed).toBe(false)
    expect(sanitizePin({ ...valid, traceAllowed: 'yes' }, 'fixture')?.traceUrl).toBeNull()
    expect(sanitizePin({ ...valid, traceAllowed: false }, 'fixture')?.traceUrl).toBeNull()
  })

  it('rejects entries that are not usable references', () => {
    expect(sanitizePin(null, 'fixture')).toBeNull()
    expect(sanitizePin('fixture-1', 'fixture')).toBeNull()
    expect(sanitizePin({ ...valid, id: '' }, 'fixture')).toBeNull()
    expect(sanitizePin({ ...valid, style: 'Watercolour' }, 'fixture')).toBeNull()
    expect(sanitizePin({ ...valid, imageUrl: 'javascript:alert(1)' }, 'fixture')).toBeNull()
    expect(sanitizePin({ ...valid, imageUrl: 'https://example.test/x.png' }, 'fixture')).toBeNull()
  })

  it('keeps fixture and live assets in their own namespaces', () => {
    expect(isLiveAssetId('ls_synthetic_ac1f55b7390698a7')).toBe(true)
    expect(isLiveAssetId('fixture-1')).toBe(false)
    // A live gallery asset must never be adopted by the fixture store.
    expect(sanitizePin({ ...valid, id: 'ls_synthetic_ac1f55b7390698a7' }, 'fixture')).toBeNull()
    // ...and a fixture asset is not a live pin either.
    expect(sanitizePin(valid, 'live')).toBeNull()
    expect(
      sanitizePin({ ...valid, id: 'ls_synthetic_ac1f55b7390698a7', imageUrl: '/api/v1/assets/x/thumbnail' }, 'live')?.id,
    ).toBe('ls_synthetic_ac1f55b7390698a7')
  })
})

describe('readLocalPins', () => {
  it('returns an empty list for missing or corrupt storage', () => {
    expect(readLocalPins('fixture')).toEqual([])
    localStorage.setItem(pinsKey('fixture'), '{not json')
    expect(readLocalPins('fixture')).toEqual([])
    localStorage.setItem(pinsKey('fixture'), '{"pins":1}')
    expect(readLocalPins('fixture')).toEqual([])
  })

  it('drops invalid entries, de-duplicates, and caps the list', () => {
    const many = Array.from({ length: MAX_LOCAL_PINS + 10 }, (_, index) => ({
      ...valid,
      id: `fixture-${index}`,
    }))
    localStorage.setItem(
      pinsKey('fixture'),
      JSON.stringify([valid, valid, { ...valid, style: 'Nope' }, null, ...many]),
    )
    const pins = readLocalPins('fixture')
    expect(pins).toHaveLength(MAX_LOCAL_PINS)
    expect(new Set(pins.map((pin) => pin.id)).size).toBe(pins.length)
  })

  it('round-trips what it wrote', () => {
    const pin = sanitizePin(valid, 'fixture') as ReferenceAsset
    writeLocalPins('fixture', [pin])
    expect(readLocalPins('fixture')).toEqual([pin])
  })
})

describe('legacy stores', () => {
  it('migrates only valid fixture entries out of the un-namespaced key', () => {
    localStorage.setItem(
      LEGACY_FIXTURE_PINS_KEY,
      JSON.stringify([
        valid,
        { ...valid, id: 'ls_synthetic_ac1f55b7390698a7' }, // a live asset: dropped
        { junk: true },
      ]),
    )
    const migrated = readLocalPins('fixture')
    expect(migrated.map((pin) => pin.id)).toEqual(['fixture-1'])
    expect(localStorage.getItem(LEGACY_FIXTURE_PINS_KEY)).toBeNull()
    // Idempotent: a second read has nothing left to migrate.
    expect(migrateLegacyFixturePins()).toBeNull()
    expect(readLocalPins('fixture').map((pin) => pin.id)).toEqual(['fixture-1'])
  })

  it('never migrates the legacy key into the live namespace', () => {
    localStorage.setItem(LEGACY_FIXTURE_PINS_KEY, JSON.stringify([valid]))
    expect(readLocalPins('live')).toEqual([])
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()
    expect(localStorage.getItem(LEGACY_FIXTURE_PINS_KEY)).not.toBeNull()
  })

  it('hands cached live pin ids to the caller once and clears them', () => {
    const live = { ...valid, id: 'ls_synthetic_ac1f55b7390698a7', imageUrl: '/api/v1/assets/x/thumbnail' }
    localStorage.setItem(pinsKey('live'), JSON.stringify([live, valid]))
    // Only live-shaped ids come back; the fixture entry is not smuggled over.
    expect(takeLegacyLivePinIds()).toEqual(['ls_synthetic_ac1f55b7390698a7'])
    expect(localStorage.getItem(pinsKey('live'))).toBeNull()
    expect(takeLegacyLivePinIds()).toEqual([])
  })
})
