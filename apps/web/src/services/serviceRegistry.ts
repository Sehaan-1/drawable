/**
 * Chooses between the live API and the local fixture services.
 *
 * `VITE_LINESCOUT_SERVICES` forces a mode (`live` | `fixture`). Otherwise the
 * app probes `/api/v1/health` once at startup: a reachable API wins, anything
 * else (connection refused, proxy error, timeout) silently keeps the fixture
 * service so the canvas remains usable offline.
 */

import { create } from 'zustand'
import type { HealthResult } from '../lib/types'
import { useSearchStore, type GalleryKind } from '../state/searchStore'
import { fixtureServices, type FrontendServices } from './frontendServices'
import { liveServices } from './liveServices'

export type ServiceMode = 'probing' | 'live' | 'fixture'

interface ServiceState {
  mode: ServiceMode
  health: HealthResult | null
  services: FrontendServices
  sessionId: string
  probe: () => Promise<void>
}

const SESSION_KEY = 'linescout-session-id'

function sessionId(): string {
  const existing = sessionStorage.getItem(SESSION_KEY)
  if (existing) return existing
  const created = crypto.randomUUID()
  sessionStorage.setItem(SESSION_KEY, created)
  return created
}

const forced = import.meta.env.VITE_LINESCOUT_SERVICES as string | undefined

export const useServiceStore = create<ServiceState>((set) => {
  const activate = (mode: ServiceMode, services: FrontendServices, health: HealthResult | null) => {
    set({ mode, services, health })
    // Pins are namespaced per gallery kind: switching between the fixture
    // and live galleries swaps the pin set so the two never mix.
    if (mode === 'live' || mode === 'fixture') {
      useSearchStore.getState().setGallery(mode as GalleryKind)
    }
  }
  return {
    mode: forced === 'fixture' ? 'fixture' : forced === 'live' ? 'live' : 'probing',
    health: null,
    services: forced === 'live' ? liveServices : fixtureServices,
    sessionId: sessionId(),
    probe: async () => {
      if (forced === 'fixture') {
        activate('fixture', fixtureServices, await fixtureServices.health.get())
        return
      }
      const controller = new AbortController()
      const timer = window.setTimeout(() => controller.abort(), 2500)
      try {
        const health = await liveServices.health.get(controller.signal)
        activate('live', liveServices, health)
      } catch {
        if (forced === 'live') {
          activate('live', liveServices, { mode: 'cpu', ready: false, message: 'API unreachable', live: true })
        } else {
          activate('fixture', fixtureServices, await fixtureServices.health.get())
        }
      } finally {
        window.clearTimeout(timer)
      }
    },
  }
})

export function currentServices(): FrontendServices {
  return useServiceStore.getState().services
}
