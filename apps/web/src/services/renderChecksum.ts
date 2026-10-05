import { rasterizeDocument } from '../lib/exportDocument'
import type { DrawingDocument } from '../lib/types'

/** Hash dimensions plus uncompressed canonical RGBA pixels, not PNG encoder bytes. */
export async function checksumRenderedDocument(document: DrawingDocument): Promise<string> {
  const canvas = await rasterizeDocument(document, false)
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Could not read the canonical rendered drawing.')
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
  const input = new Uint8Array(8 + pixels.length)
  const dimensions = new DataView(input.buffer)
  dimensions.setUint32(0, canvas.width, false)
  dimensions.setUint32(4, canvas.height, false)
  input.set(pixels, 8)
  const digest = await crypto.subtle.digest('SHA-256', input)
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
}
