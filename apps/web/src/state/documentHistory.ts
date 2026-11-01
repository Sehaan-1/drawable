import type { DrawingDocument, DrawingLayer, DrawingOperation, TraceState } from '../lib/types'

/**
 * Reversible history entries + structural-sharing document helpers.
 *
 * Invariants (see plan §3.1):
 *  - Documents reachable from the store are immutable: every helper returns a new
 *    object and reuses untouched subobjects (layers, operations arrays, operations).
 *  - Operations and their `points` arrays are immutable after commit.
 *  - History entries retain only the payload required to undo/redo a single edit,
 *    never a full document snapshot.
 */

type LayerSettings = Pick<DrawingLayer, 'name' | 'visible' | 'opacity'>

type TraceField = 'assetId' | 'imageUrl' | 'visible' | 'opacity' | 'scale'
type TraceSettings = Partial<Pick<TraceState, TraceField>>

export interface HistoryLimits {
  maxEntries: number
  maxBytes: number
}

export const DEFAULT_HISTORY_LIMITS: HistoryLimits = {
  maxEntries: 500,
  maxBytes: 32 * 1024 * 1024,
}

interface HistoryEntryBase {
  bytes: number
}

interface OperationAppendedPayload {
  kind: 'operation_appended'
  layerId: string
  operation: DrawingOperation
}

interface LayerPropertiesChangedPayload {
  kind: 'layer_properties_changed'
  layerId: string
  before: LayerSettings
  after: LayerSettings
}

interface LayerMovedPayload {
  kind: 'layer_moved'
  layerId: string
  from: number
  to: number
}

interface LayerClearedPayload {
  kind: 'layer_cleared'
  layerId: string
  operations: DrawingOperation[]
}

interface TitleChangedPayload {
  kind: 'title_changed'
  before: string
  after: string
}

interface TraceChangedPayload {
  kind: 'trace_changed'
  // Only the fields the command actually wrote. `imageUrl` is included for
  // setTrace/clearTrace and deliberately excluded from updateTrace so a later
  // setResolvedTraceImage() is not clobbered by redo.
  before: TraceSettings
  after: TraceSettings
}

export type HistoryPayload =
  | OperationAppendedPayload
  | LayerPropertiesChangedPayload
  | LayerMovedPayload
  | LayerClearedPayload
  | TitleChangedPayload
  | TraceChangedPayload

export type HistoryEntry = HistoryEntryBase & HistoryPayload

// ── Size estimation ──────────────────────────────────────────────────────────

const POINT_BYTES = 48 // conservative: 4 numbers plus per-point object overhead
const OPERATION_OVERHEAD = 96
const ENTRY_OVERHEAD = 128
const TITLE_OVERHEAD = 64

export function estimateOperationBytes(operation: DrawingOperation): number {
  if (operation.kind === 'stroke') {
    return OPERATION_OVERHEAD + operation.points.length * POINT_BYTES
  }
  return OPERATION_OVERHEAD + 64
}

function estimatePayloadBytes(payload: HistoryPayload): number {
  switch (payload.kind) {
    case 'operation_appended':
      return estimateOperationBytes(payload.operation)
    case 'layer_cleared':
      return payload.operations.reduce((sum, operation) => sum + estimateOperationBytes(operation), 0)
    case 'layer_properties_changed':
      return ENTRY_OVERHEAD + 2 * (payload.before.name.length + payload.after.name.length)
    case 'layer_moved':
      return ENTRY_OVERHEAD
    case 'title_changed':
      return TITLE_OVERHEAD + 2 * (payload.before.length + payload.after.length)
    case 'trace_changed':
      return ENTRY_OVERHEAD + 2 * (
        (payload.before.imageUrl?.length ?? 0)
        + (payload.after.imageUrl?.length ?? 0)
        + (payload.before.assetId?.length ?? 0)
        + (payload.after.assetId?.length ?? 0)
      )
    default: {
      const exhaustive: never = payload
      return exhaustive
    }
  }
}

/** Wrap a payload with its conservative byte estimate. */
export function createHistoryEntry(payload: HistoryPayload): HistoryEntry {
  return { ...payload, bytes: estimatePayloadBytes(payload) }
}

// ── Structural-sharing primitives ────────────────────────────────────────────

export function findLayer(document: DrawingDocument, layerId: string): DrawingLayer | undefined {
  return document.layers.find((layer) => layer.id === layerId)
}

/**
 * Replaces one layer. Copies only the `layers` array and the affected layer;
 * every other layer (and every operation object) keeps its identity.
 */
function replaceLayer(
  document: DrawingDocument,
  layerId: string,
  update: (layer: DrawingLayer) => DrawingLayer,
): DrawingDocument {
  const index = document.layers.findIndex((layer) => layer.id === layerId)
  const current = index >= 0 ? document.layers[index] : undefined
  if (!current) return document
  const next = update(current)
  if (next === current) return document
  const layers = document.layers.slice()
  layers[index] = next
  return { ...document, layers }
}

function moveLayerTo(document: DrawingDocument, layerId: string, target: number): DrawingDocument {
  const index = document.layers.findIndex((layer) => layer.id === layerId)
  if (index < 0 || target < 0 || target >= document.layers.length || index === target) return document
  const layers = document.layers.slice()
  const [layer] = layers.splice(index, 1)
  if (!layer) return document
  layers.splice(target, 0, layer)
  return { ...document, layers }
}

// ── Apply ────────────────────────────────────────────────────────────────────

export function applyHistoryEntry(
  document: DrawingDocument,
  entry: HistoryEntry,
  direction: 'undo' | 'redo',
): DrawingDocument {
  switch (entry.kind) {
    case 'operation_appended': {
      const layer = findLayer(document, entry.layerId)
      if (!layer) return document
      if (direction === 'redo') {
        if (layer.operations.at(-1)?.id === entry.operation.id) return document
        return replaceLayer(document, entry.layerId, (current) => ({
          ...current,
          operations: [...current.operations, entry.operation],
        }))
      }
      const operations = layer.operations.at(-1)?.id === entry.operation.id
        ? layer.operations.slice(0, -1)
        : layer.operations.filter((operation) => operation.id !== entry.operation.id)
      return replaceLayer(document, entry.layerId, (current) => ({ ...current, operations }))
    }
    case 'layer_properties_changed':
      return replaceLayer(document, entry.layerId, (layer) => ({
        ...layer,
        ...(direction === 'redo' ? entry.after : entry.before),
      }))
    case 'layer_moved':
      return moveLayerTo(document, entry.layerId, direction === 'redo' ? entry.to : entry.from)
    case 'layer_cleared':
      return replaceLayer(document, entry.layerId, (layer) => ({
        ...layer,
        operations: direction === 'redo' ? [] : entry.operations,
      }))
    case 'title_changed':
      return { ...document, title: direction === 'redo' ? entry.after : entry.before }
    case 'trace_changed':
      return {
        ...document,
        trace: { ...document.trace, ...(direction === 'redo' ? entry.after : entry.before) },
      }
    default: {
      const exhaustive: never = entry
      return exhaustive
    }
  }
}

// ── Trimming ─────────────────────────────────────────────────────────────────

export function trimPast(entries: HistoryEntry[], limits: HistoryLimits): HistoryEntry[] {
  let total = entries.reduce((sum, entry) => sum + entry.bytes, 0)
  let start = 0
  // Keep at least the newest entry even if it alone exceeds maxBytes.
  while (entries.length - start > limits.maxEntries || (total > limits.maxBytes && entries.length - start > 1)) {
    total -= entries[start]!.bytes
    start += 1
  }
  return start === 0 ? entries : entries.slice(start)
}

export function trimFuture(entries: HistoryEntry[], limits: HistoryLimits): HistoryEntry[] {
  let total = entries.reduce((sum, entry) => sum + entry.bytes, 0)
  let end = entries.length
  // Future is newest-first; drop from the end (oldest redo entries).
  while (end > limits.maxEntries || (total > limits.maxBytes && end > 1)) {
    end -= 1
    total -= entries[end]!.bytes
  }
  return end === entries.length ? entries : entries.slice(0, end)
}
