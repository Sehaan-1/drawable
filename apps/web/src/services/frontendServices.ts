import type { EventRequest, PreferencesResponse, PreferencesUpdate } from '@drawable/contracts'
import { fixtureAssets, fixtureHealth, fixtureSearch } from './fixtures'
import { readLocalPins, writeLocalPins } from './pinStorage'
import type { HealthResult, ReferenceAsset, SearchRequest, SearchResponse } from '../lib/types'
import { traceSource } from '../lib/trace'

export interface SearchClient {
  search(request: SearchRequest, signal: AbortSignal): Promise<SearchResponse>
}

export interface AssetClient {
  /**
   * The trace-layer image for a saved reference, or `null` when the asset is
   * gone or its source permissions no longer allow tracing. Called when a
   * document is reopened, so a revoked permission cannot be resurrected from
   * a stale local document.
   */
  resolveTrace(assetId: string, signal?: AbortSignal): Promise<string | null>
}

/** A pin dropped by the server because its asset is no longer eligible. */
export interface RevokedPinView {
  assetId: string
  reasons: string[]
}

export interface PinSnapshot {
  pins: ReferenceAsset[]
  revoked: RevokedPinView[]
}

/**
 * Durable pin state for the active gallery.
 *
 * Every call answers with the whole set so the UI can replace its view
 * atomically instead of inferring what changed — which also means a
 * revalidation that dropped a pin is visible immediately.
 */
export interface PinClient {
  list(): Promise<PinSnapshot>
  pin(asset: ReferenceAsset): Promise<PinSnapshot>
  unpin(assetId: string): Promise<PinSnapshot>
}

export interface FrontendServices {
  health: { get(signal?: AbortSignal): Promise<HealthResult> }
  search: SearchClient
  assets: AssetClient
  events: { record(event: EventRequest): Promise<void> }
  preferences: {
    get(): Promise<PreferencesResponse | null>
    update(update: PreferencesUpdate): Promise<PreferencesResponse | null>
  }
  pins: PinClient
  curation?: Record<string, never>
  benchmarks?: Record<string, never>
}

/**
 * Offline pin store for the fixture gallery.
 *
 * Fixture assets are procedural and local, so their pins are kept in
 * `localStorage` — a namespace the live gallery never reads or writes.
 */
export function localPinClient(): PinClient {
  const snapshot = (pins: ReferenceAsset[]): PinSnapshot => ({ pins, revoked: [] })
  const read = () => readLocalPins('fixture')
  return {
    list: async () => snapshot(read()),
    pin: async (asset) => {
      const existing = read().filter((item) => item.id !== asset.id)
      const pins = [{ ...asset, pinnedAt: new Date().toISOString() }, ...existing]
      writeLocalPins('fixture', pins)
      return snapshot(pins)
    },
    unpin: async (assetId) => {
      const pins = read().filter((item) => item.id !== assetId)
      writeLocalPins('fixture', pins)
      return snapshot(pins)
    },
  }
}

export const fixtureServices: FrontendServices = {
  health: { get: async () => fixtureHealth },
  search: { search: fixtureSearch },
  assets: {
    resolveTrace: async (assetId) => {
      const asset = fixtureAssets.find((item) => item.id === assetId)
      return asset ? traceSource(asset) : null
    },
  },
  events: { record: async () => undefined },
  preferences: { get: async () => null, update: async () => null },
  pins: localPinClient(),
  curation: {},
  benchmarks: {},
}
