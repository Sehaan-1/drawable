/**
 * Local pin persistence for the **fixture** gallery.
 *
 * Pins are durable application state, and the two galleries are kept in
 * separate stores that can never be read into each other:
 *
 * * **live** pins are owned by the API and live in its SQLite database
 *   (`/api/v1/pins`), so they survive a page reload *and* an API restart, and
 *   are revalidated against the current gallery on every read;
 * * **fixture** pins are offline-only and are kept here, in `localStorage`
 *   under `drawable-pins:fixture:<version>`.
 *
 * Everything read from storage is untrusted input: entries are validated and
 * normalised, unusable ones are dropped, and permission-bearing fields fail
 * closed (an entry that does not explicitly say tracing is allowed is treated
 * as not traceable).
 */

import type { GalleryKind } from '@drawable/contracts'
import type { ReferenceAsset, ReferenceStyle } from '../lib/types'

export const PINS_VERSION = 1
/** Pre-namespacing key. It only ever held *fixture* pins. */
export const LEGACY_FIXTURE_PINS_KEY = 'drawable-fixture-pins'
/** Defensive cap so a runaway writer cannot fill the origin's quota. */
export const MAX_LOCAL_PINS = 200

const STYLES: ReferenceStyle[] = ['Manga / anime', 'Western ink', 'Realistic', 'Cartoon', 'Gesture']
const MATCHES: ReferenceAsset['match'][] = ['Strong', 'Close', 'Related']

export function pinsKey(kind: GalleryKind): string {
  return `drawable-pins:${kind}:${PINS_VERSION}`
}

/**
 * Live gallery assets are identified by the manifest's `ls_…` ids. A fixture
 * store must never adopt one: it would put a real-gallery asset — with real
 * permissions — into the offline namespace, where none of that is checked.
 */
export function isLiveAssetId(id: string): boolean {
  return id.startsWith('ls_')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function text(value: unknown, max = 400): string | null {
  return typeof value === 'string' && value.length > 0 && value.length <= max ? value : null
}

/** Only inert, same-origin-ish image sources survive: no `javascript:` etc. */
function imageSource(value: unknown): string | null {
  const raw = text(value, 2_000_000)
  if (!raw) return null
  if (raw.startsWith('data:image/')) return raw
  if (raw.startsWith('/')) return raw
  return null
}

/**
 * Validate one stored entry for `kind`, or return `null` to drop it.
 *
 * `traceAllowed` is only honoured when the entry says so explicitly, and the
 * trace source is derived from it — a pin can never be traced just because it
 * happens to carry an image URL.
 */
export function sanitizePin(value: unknown, kind: GalleryKind): ReferenceAsset | null {
  if (!isRecord(value)) return null
  const id = text(value.id, 128)
  if (!id) return null
  if (kind === 'fixture' && isLiveAssetId(id)) return null
  if (kind === 'live' && !isLiveAssetId(id)) return null
  const imageUrl = imageSource(value.imageUrl)
  if (!imageUrl) return null
  const style = STYLES.find((candidate) => candidate === value.style)
  if (!style) return null
  const traceAllowed = value.traceAllowed === true
  const fullImageUrl = imageSource(value.fullImageUrl) ?? undefined
  return {
    id,
    title: text(value.title, 200) ?? id,
    imageUrl,
    fullImageUrl,
    style,
    scope: text(value.scope, 200) ?? 'Reference',
    source: text(value.source, 200) ?? 'Local fixture',
    native: value.native === true,
    match: MATCHES.find((candidate) => candidate === value.match) ?? 'Related',
    relevance: typeof value.relevance === 'number' && Number.isFinite(value.relevance) ? value.relevance : undefined,
    traceAllowed,
    // Derived, never trusted from storage: no permission, no trace source.
    traceUrl: traceAllowed ? (fullImageUrl ?? imageUrl) : null,
  }
}

export function sanitizePins(raw: unknown, kind: GalleryKind): ReferenceAsset[] {
  if (!Array.isArray(raw)) return []
  const seen = new Set<string>()
  const pins: ReferenceAsset[] = []
  for (const entry of raw) {
    const pin = sanitizePin(entry, kind)
    if (!pin || seen.has(pin.id)) continue
    seen.add(pin.id)
    pins.push(pin)
    if (pins.length >= MAX_LOCAL_PINS) break
  }
  return pins
}

function parse(rawText: string | null, kind: GalleryKind): ReferenceAsset[] {
  if (rawText === null) return []
  try {
    return sanitizePins(JSON.parse(rawText), kind)
  } catch {
    return []
  }
}

/**
 * Fold the pre-namespacing key into the fixture namespace, exactly once.
 *
 * Only entries that validate as fixture assets are carried over, so a
 * corrupted or hand-edited legacy blob — or one that somehow contains live
 * `ls_…` assets — cannot smuggle anything into the fixture store.
 */
export function migrateLegacyFixturePins(): ReferenceAsset[] | null {
  const legacy = localStorage.getItem(LEGACY_FIXTURE_PINS_KEY)
  if (legacy === null) return null
  const migrated = parse(legacy, 'fixture')
  localStorage.setItem(pinsKey('fixture'), JSON.stringify(migrated))
  localStorage.removeItem(LEGACY_FIXTURE_PINS_KEY)
  return migrated
}

export function readLocalPins(kind: GalleryKind): ReferenceAsset[] {
  if (kind === 'fixture' && localStorage.getItem(pinsKey('fixture')) === null) {
    const migrated = migrateLegacyFixturePins()
    if (migrated !== null) return migrated
  }
  return parse(localStorage.getItem(pinsKey(kind)), kind)
}

export function writeLocalPins(kind: GalleryKind, pins: ReferenceAsset[]): void {
  try {
    localStorage.setItem(pinsKey(kind), JSON.stringify(pins.slice(0, MAX_LOCAL_PINS)))
  } catch {
    // A full or unavailable quota must never break pinning in the session.
  }
}

/**
 * Read and clear locally cached **live** pins written by older builds.
 *
 * Live pins are server state now; the ids are handed to the API so it can
 * revalidate each one (an asset that lost permission is simply not re-pinned)
 * and the local copy is dropped either way.
 */
export function takeLegacyLivePinIds(): string[] {
  const key = pinsKey('live')
  const stored = localStorage.getItem(key)
  if (stored === null) return []
  const ids = parse(stored, 'live').map((pin) => pin.id)
  localStorage.removeItem(key)
  return ids
}
