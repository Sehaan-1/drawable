import type { DrawingDocument, StoredRasterAsset } from '../lib/types'
import { loadRasterAsset } from './persistence'

/**
 * How much of the document a caller needs its imported artwork for.
 *
 * Scope is per-caller on purpose:
 *
 * * `'visible'` is for render paths — canvas, thumbnails, PNG/SVG export, and
 *   the search snapshot. Those only ever *show* visible layers, so a missing
 *   blob on a hidden layer must not be reported as an omission from a picture
 *   that would never have contained it.
 * * `'all'` is for the durable export — the `.drawable` project — where a
 *   hidden layer's artwork still has to be in the file so it survives
 *   unhiding after a round-trip.
 */
export type RasterScope = 'visible' | 'all'

/** What resolving one imported asset comes to. Never a rejection: a gone blob is data, not an exception. */
type RasterImageOutcome = { image: HTMLImageElement } | { unavailable: true }

const imageCache = new Map<string, Promise<RasterImageOutcome>>()

function referencedAssetIds(document: DrawingDocument, scope: RasterScope) {
  const layers = scope === 'visible' ? document.layers.filter((layer) => layer.visible && layer.opacity > 0) : document.layers
  return [...new Set(layers.flatMap((layer) => layer.operations.flatMap((operation) => operation.kind === 'raster' ? [operation.assetId] : [])))]
}

function decodeBlob(blob: Blob) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const url = URL.createObjectURL(blob)
    const image = new Image()
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(image)
    }
    image.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Imported artwork could not be decoded.'))
    }
    image.src = url
  })
}

async function loadImage(assetId: string): Promise<RasterImageOutcome> {
  try {
    const asset = await loadRasterAsset(assetId)
    if (!asset) return { unavailable: true }
    return { image: await decodeBlob(asset.blob) }
  } catch {
    // A storage error or an undecodable blob degrades to an omission, exactly
    // like a blob that was never stored.
    return { unavailable: true }
  }
}

function cachedImage(assetId: string) {
  let pending = imageCache.get(assetId)
  if (!pending) {
    pending = loadImage(assetId)
    imageCache.set(assetId, pending)
    // Don't cache the failure. The promise is stored before its outcome is
    // known so concurrent callers share one read — but if the outcome is
    // "unavailable", one transient storage error would otherwise latch the
    // import as absent for canvas, exports, *and* search until reload. Only a
    // usable image stays cached; an omission is revoked so the next render
    // retries storage.
    void pending.then((value) => {
      if ('unavailable' in value && imageCache.get(assetId) === pending) imageCache.delete(assetId)
    })
  }
  return pending
}

export interface RasterImageResolution {
  images: Map<string, CanvasImageSource>
  /** Referenced asset ids whose stored blob is gone (or unreadable). */
  unavailable: string[]
}

export async function resolveRasterImages(document: DrawingDocument, scope: RasterScope = 'visible'): Promise<RasterImageResolution> {
  const images = new Map<string, CanvasImageSource>()
  const unavailable: string[] = []
  await Promise.all(referencedAssetIds(document, scope).map(async (assetId) => {
    const outcome = await cachedImage(assetId)
    if ('unavailable' in outcome) unavailable.push(assetId)
    else images.set(assetId, outcome.image)
  }))
  return { images, unavailable }
}

export interface RasterBitmapResolution {
  bitmaps: Array<{ id: string; bitmap: ImageBitmap }>
  unavailable: string[]
}

export async function resolveRasterBitmaps(document: DrawingDocument, scope: RasterScope = 'visible'): Promise<RasterBitmapResolution> {
  const bitmaps: Array<{ id: string; bitmap: ImageBitmap }> = []
  const unavailable: string[] = []
  for (const assetId of referencedAssetIds(document, scope)) {
    const asset = await loadRasterAsset(assetId).catch(() => null)
    if (!asset) {
      unavailable.push(assetId)
      continue
    }
    try {
      bitmaps.push({ id: assetId, bitmap: await createImageBitmap(asset.blob) })
    } catch {
      unavailable.push(assetId)
    }
  }
  return { bitmaps, unavailable }
}

export interface StoredAssetResolution {
  assets: StoredRasterAsset[]
  unavailable: string[]
}

export async function loadReferencedRasterAssets(document: DrawingDocument, scope: RasterScope = 'visible'): Promise<StoredAssetResolution> {
  const assets: StoredRasterAsset[] = []
  const unavailable: string[] = []
  for (const assetId of referencedAssetIds(document, scope)) {
    const asset = await loadRasterAsset(assetId).catch(() => null)
    if (asset) assets.push(asset)
    else unavailable.push(assetId)
  }
  return { assets, unavailable }
}

/**
 * The one sentence every surface reuses when artwork turns out to be gone, so
 * the dialog, the export notice, and the search warning never drift apart.
 */
export function unavailableArtworkSummary(unavailable: readonly string[]) {
  return unavailable.length === 1
    ? 'One imported artwork is no longer stored on this device'
    : `${unavailable.length} imported artworks are no longer stored on this device`
}

export function clearRasterImageCache() {
  imageCache.clear()
}
