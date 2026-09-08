/**
 * Trace-layer permission, in one place.
 *
 * Whether a reference may be placed on the trace layer is decided by the
 * asset's recorded *source permission* (`traceAllowed`), never by whether its
 * line art is native or extracted. The UI asks this module rather than
 * reading URLs directly, so every entry point — card button, detail button,
 * keyboard path, restored document — enforces the same rule.
 */

import type { ReferenceAsset } from './types'

/** The image the trace layer may load, or `null` when tracing is forbidden. */
export function traceSource(asset: Pick<ReferenceAsset, 'traceAllowed' | 'traceUrl'>): string | null {
  if (!asset.traceAllowed) return null
  return asset.traceUrl ?? null
}

export function canTrace(asset: Pick<ReferenceAsset, 'traceAllowed' | 'traceUrl'>): boolean {
  return traceSource(asset) !== null
}
