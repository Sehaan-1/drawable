/**
 * Document history performance guard.
 *
 * Measures the shape of the history cost, not absolute wall time:
 *  - retained history payload must track the committed stroke payload,
 *  - untouched layers must be reused by reference,
 *  - a commit must not deep-clone the document.
 *
 * Run with `npm --workspace=@drawable/web run test:perf`.
 */
import { describe, expect, it } from 'vitest'
import { makeStrokeOperation } from '../lib/drawing'
import type { StrokeOperation } from '../lib/types'
import { createDocumentStore } from './documentStore'
import { estimateOperationBytes } from './documentHistory'

const STROKES = 200
const POINTS_PER_STROKE = 500

function longStroke(index: number): StrokeOperation {
  const points = Array.from({ length: POINTS_PER_STROKE }, (_, point) => ({
    x: (index * 17 + point) % 2048,
    y: (index * 29 + point * 3) % 2048,
    pressure: 0.5,
    time: point,
  }))
  return makeStrokeOperation('pressure', points, 12, 50, false)
}

function percentile(values: number[], fraction: number) {
  if (values.length === 0) return 0
  const sorted = values.slice().sort((left, right) => left - right)
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]!
}

describe('document history performance', () => {
  it('keeps retained history proportional to stroke payload', () => {
    const store = createDocumentStore()
    const initialLayers = store.getState().document.layers
    const commitMs: number[] = []
    const strokes: StrokeOperation[] = []

    for (let index = 0; index < STROKES; index += 1) {
      const operation = longStroke(index)
      strokes.push(operation)
      const started = performance.now()
      store.getState().commitOperation(operation)
      commitMs.push(performance.now() - started)
    }

    const { document, past } = store.getState()
    const retainedBytes = past.reduce((sum, entry) => sum + entry.bytes, 0)
    const rawBytes = STROKES * estimateOperationBytes(longStroke(0))

    const finalCloneStarted = performance.now()
    structuredClone(document)
    const finalCloneMs = performance.now() - finalCloneStarted

    const totalMs = commitMs.reduce((sum, value) => sum + value, 0)
    console.info([
      `[history perf] commits=${STROKES} points/stroke=${POINTS_PER_STROKE}`,
      `p50=${percentile(commitMs, 0.5).toFixed(3)}ms`,
      `p95=${percentile(commitMs, 0.95).toFixed(3)}ms`,
      `total=${totalMs.toFixed(1)}ms`,
      `retainedBytes=${retainedBytes} rawBytes=${rawBytes} ratio=${(retainedBytes / rawBytes).toFixed(2)}`,
      `finalCloneMs=${finalCloneMs.toFixed(2)}ms (old per-commit clone ≈ ${(finalCloneMs * STROKES).toFixed(0)}ms of allocation work)`,
    ].join(' '))

    // Shape assertions (deterministic).
    expect(past).toHaveLength(STROKES)
    expect(retainedBytes).toBeLessThanOrEqual(rawBytes * 2)
    expect(document.layers[1]).toBe(initialLayers[1])
    expect(document.layers[2]).toBe(initialLayers[2])
    expect(document.layers[3]).toBe(initialLayers[3])

    const firstEntry = past[0]
    expect(firstEntry?.kind).toBe('operation_appended')
    if (firstEntry?.kind !== 'operation_appended' || firstEntry.operation.kind !== 'stroke') {
      throw new Error('Expected a stroke operation entry')
    }
    // The store shallow-copies the operation object and shares the points array,
    // and the same entry object is what undo/redo append and remove.
    const firstStroke = strokes[0]!
    expect(firstEntry.operation.id).toBe(firstStroke.id)
    expect(firstEntry.operation).not.toBe(firstStroke)
    const storedOperation = document.layers[0]!.operations[0]!
    if (storedOperation.kind !== 'stroke') throw new Error('Expected a stroke operation')
    expect(firstEntry.operation.points).toBe(firstStroke.points)
    expect(storedOperation.points).toBe(firstStroke.points)
    expect(storedOperation).toBe(firstEntry.operation)

    // Generous ceiling only; the byte/reference assertions above are the real guard.
    expect(totalMs).toBeLessThan(5000)
  }, 20_000)
})
