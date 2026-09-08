/// <reference lib="webworker" />
import { getStroke } from 'perfect-freehand'
import { measureInk, type RasterInk } from '../lib/rasterInk'
import type { DrawingDocument, StrokeOperation } from '../lib/types'
import { LOGICAL_SIZE } from '../lib/types'
import { outlineToPath } from '../lib/drawing'

const SNAPSHOT_SIZE = 512

interface SnapshotMessage {
  id: string
  /** Request ownership identity, echoed back verbatim so the caller can gate on it. */
  token: number
  documentId: string
  revision: number
  generation: number
  document: DrawingDocument
  rasterAssets: Array<{ id: string; bitmap: ImageBitmap }>
}

interface CancelMessage {
  cancel: string
}

type WorkerMessage = SnapshotMessage | CancelMessage

function isCancel(message: WorkerMessage): message is CancelMessage {
  return typeof (message as CancelMessage).cancel === 'string'
}

/** Requests cancelled before the worker got to them; their bitmaps are still closed. */
const cancelled = new Set<string>()

function drawOperation(context: OffscreenCanvasRenderingContext2D, operation: StrokeOperation) {
  const points = getStroke(operation.points.map((point) => [point.x, point.y, point.pressure]), {
    size: operation.size,
    thinning: operation.tool === 'pressure' ? 0.65 : 0,
    smoothing: operation.smoothing,
    streamline: operation.streamline,
    simulatePressure: operation.simulatePressure,
    start: { cap: true, taper: 0 },
    end: { cap: operation.tool !== 'pressure', taper: operation.tool === 'pressure' ? 8 : 0 },
    last: true,
  })
  const path = outlineToPath(points)
  if (!path) return
  context.save()
  context.fillStyle = '#111214'
  context.globalCompositeOperation = operation.tool === 'eraser' ? 'destination-out' : 'source-over'
  context.fill(new Path2D(path))
  context.restore()
}

/**
 * Render the search snapshot and measure its ink.
 *
 * Both halves are keyed to the request identity that produced them: the
 * measurement is only trustworthy for the canvas it was taken from, and the
 * echo is what lets the caller discard a snapshot whose request has already
 * been superseded. A cancelled request is dropped before the next layer is
 * drawn, so an abort mid-snapshot stops costing GPU/canvas time instead of
 * finishing a picture nobody will read.
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data
  if (isCancel(message)) {
    cancelled.add(message.cancel)
    return
  }
  const { id, token, documentId, revision, generation, document, rasterAssets } = message
  const assets = new Map(rasterAssets.map((asset) => [asset.id, asset.bitmap]))
  const abort = () => cancelled.delete(id)
  try {
    if (cancelled.has(id)) return
    const output = new OffscreenCanvas(SNAPSHOT_SIZE, SNAPSHOT_SIZE)
    const outputContext = output.getContext('2d')
    if (!outputContext) throw new Error('2D worker canvas unavailable')
    outputContext.fillStyle = '#fff'
    outputContext.fillRect(0, 0, SNAPSHOT_SIZE, SNAPSHOT_SIZE)
    for (const layer of [...document.layers].reverse()) {
      if (cancelled.has(id)) return
      if (!layer.visible || layer.opacity <= 0) continue
      const layerCanvas = new OffscreenCanvas(SNAPSHOT_SIZE, SNAPSHOT_SIZE)
      const layerContext = layerCanvas.getContext('2d')
      if (!layerContext) continue
      layerContext.scale(SNAPSHOT_SIZE / LOGICAL_SIZE, SNAPSHOT_SIZE / LOGICAL_SIZE)
      for (const operation of layer.operations) {
        if (operation.kind === 'raster') {
          const bitmap = assets.get(operation.assetId)
          if (bitmap) layerContext.drawImage(bitmap, operation.x, operation.y, operation.width, operation.height)
        } else drawOperation(layerContext, operation)
      }
      outputContext.globalAlpha = layer.opacity
      outputContext.drawImage(layerCanvas, 0, 0)
    }
    outputContext.globalAlpha = 1
    if (cancelled.has(id)) return
    const ink: RasterInk | null = (() => {
      try {
        const pixels = outputContext.getImageData(0, 0, SNAPSHOT_SIZE, SNAPSHOT_SIZE)
        return measureInk(pixels.data, pixels.width, pixels.height)
      } catch {
        return null
      }
    })()
    const image = await output.convertToBlob({ type: 'image/png' })
    self.postMessage({ id, token, documentId, revision, generation, image, ink })
  } catch (error) {
    self.postMessage({ id, token, documentId, revision, generation, error: error instanceof Error ? error.message : 'Snapshot failed' })
  } finally {
    for (const asset of rasterAssets) asset.bitmap.close()
    abort()
  }
}

export {}
