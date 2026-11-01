import { beforeEach, describe, expect, it } from 'vitest'
import { makeStrokeOperation } from '../lib/drawing'
import { createDocumentStore, useDocumentStore } from './documentStore'

describe('document store', () => {
  beforeEach(() => useDocumentStore.getState().newDocument())

  it('starts with exactly four visible drawing layers', () => {
    const { document, activeLayerId } = useDocumentStore.getState()
    expect(document.layers).toHaveLength(4)
    expect(document.layers.every((layer) => layer.visible && layer.opacity === 1)).toBe(true)
    expect(activeLayerId).toBe('layer-1')
  })

  it('commits, undoes, and redoes one stroke without losing it', () => {
    const operation = makeStrokeOperation('pressure', [
      { x: 10, y: 10, pressure: 0.4, time: 0 },
      { x: 100, y: 90, pressure: 0.8, time: 16 },
    ], 8, 50, false)
    useDocumentStore.getState().commitOperation(operation)
    expect(useDocumentStore.getState().document.layers[0]?.operations).toHaveLength(1)
    useDocumentStore.getState().undo()
    expect(useDocumentStore.getState().document.layers[0]?.operations).toHaveLength(0)
    useDocumentStore.getState().redo()
    expect(useDocumentStore.getState().document.layers[0]?.operations[0]?.id).toBe(operation.id)
  })

  it('keeps layer edits undoable and bounded', () => {
    useDocumentStore.getState().updateLayer('layer-1', { opacity: 0.35, visible: false })
    expect(useDocumentStore.getState().document.layers[0]).toMatchObject({ opacity: 0.35, visible: false })
    useDocumentStore.getState().undo()
    expect(useDocumentStore.getState().document.layers[0]).toMatchObject({ opacity: 1, visible: true })
  })

  it('keeps search revisions monotonic while excluding metadata-only edits', () => {
    const initialRevision = useDocumentStore.getState().document.revision
    useDocumentStore.getState().setTitle('Character study')
    expect(useDocumentStore.getState().document.revision).toBe(initialRevision)
    useDocumentStore.getState().updateLayer('layer-1', { name: 'Inks' })
    expect(useDocumentStore.getState().document.revision).toBe(initialRevision)
    useDocumentStore.getState().updateLayer('layer-1', { opacity: 0.5 })
    const contentRevision = useDocumentStore.getState().document.revision
    expect(contentRevision).toBeGreaterThan(initialRevision)
    useDocumentStore.getState().undo()
    expect(useDocumentStore.getState().document.revision).toBeGreaterThan(contentRevision)
  })

  it('emits recorder commands only for successful mutations', () => {
    const commands: string[] = []
    const store = createDocumentStore({ onCommand: (command) => commands.push(command.type) })
    store.getState().updateLayer('layer-1', { opacity: 1 })
    store.getState().moveLayer('layer-1', -1)
    store.getState().clearLayer('layer-1')
    store.getState().undo()
    expect(commands).toEqual([])
    store.getState().commitOperation(makeStrokeOperation('monoline', [{ x: 1, y: 1, pressure: 0.5, time: 0 }], 4, 0, false))
    expect(commands).toEqual(['operation_committed'])
  })
})

// ─── Structural sharing ──────────────────────────────────────────────────────

function makeStore(history?: { maxEntries?: number; maxBytes?: number }) {
  return createDocumentStore({ history })
}

function addStroke(store: ReturnType<typeof createDocumentStore>, points = 1) {
  const operation = makeStrokeOperation('pressure', Array.from({ length: points }, (_, index) => ({ x: index, y: index, pressure: 0.5, time: index })), 8, 50, false)
  store.getState().commitOperation(operation)
  return operation
}

describe('document store structural sharing', () => {
  it('commits an operation without reallocating untouched layers or prior operations', () => {
    const store = makeStore()
    const before = store.getState().document
    addStroke(store)
    const after = store.getState().document

    expect(after).not.toBe(before)
    expect(after.layers[0]).not.toBe(before.layers[0])
    expect(after.layers[1]).toBe(before.layers[1])
    expect(after.layers[2]).toBe(before.layers[2])
    expect(after.layers[3]).toBe(before.layers[3])

    const first = after.layers[0]!.operations[0]!
    addStroke(store)
    expect(store.getState().document.layers[0]!.operations[0]).toBe(first)
  })

  it('keeps the layers array reference for metadata-only changes', () => {
    const store = makeStore()
    addStroke(store)
    const layers = store.getState().document.layers

    store.getState().setTitle('Character study')
    expect(store.getState().document.layers).toBe(layers)

    store.getState().updateTrace({ opacity: 0.5 })
    expect(store.getState().document.layers).toBe(layers)

    store.getState().setTrace('asset-1', 'blob:trace')
    expect(store.getState().document.layers).toBe(layers)

    store.getState().setResolvedTraceImage('blob:resolved')
    expect(store.getState().document.layers).toBe(layers)
  })

  it('layer edits preserve sibling layers by reference', () => {
    const store = makeStore()
    const before = store.getState().document
    store.getState().updateLayer('layer-2', { name: 'Inks', opacity: 0.5 })
    const after = store.getState().document
    expect(after.layers[0]).toBe(before.layers[0])
    expect(after.layers[1]).not.toBe(before.layers[1])
    expect(after.layers[2]).toBe(before.layers[2])
    expect(after.layers[3]).toBe(before.layers[3])
  })

  it('does not mutate the pre-action document or committed operations', () => {
    const store = makeStore()
    const snapshot = structuredClone(store.getState().document)
    const operation = makeStrokeOperation('pressure', [{ x: 1, y: 1, pressure: 0.5, time: 0 }, { x: 2, y: 2, pressure: 0.5, time: 1 }], 8, 50, false)
    const operationSnapshot = structuredClone(operation)

    store.getState().commitOperation(operation)
    store.getState().updateLayer('layer-2', { opacity: 0.2 })
    store.getState().moveLayer('layer-2', 1)
    store.getState().setTitle('Renamed')

    expect(snapshot.layers.every((layer) => layer.operations.length === 0)).toBe(true)
    expect(snapshot.title).toBe('Untitled drawing')
    expect(snapshot.trace).toEqual({ assetId: null, imageUrl: null, visible: true, opacity: 0.3, scale: 1 })

    const committed = store.getState().document.layers[0]!.operations[0]!
    addStroke(store)
    expect(committed).toEqual(operationSnapshot)
    if (committed.kind !== 'stroke') throw new Error('Expected a stroke operation')
    expect(committed.points).toEqual(operationSnapshot.points)
  })

  it('retains history payload in proportion to the strokes, not the document', () => {
    const store = makeStore()
    const operation = makeStrokeOperation('pressure', Array.from({ length: 100 }, (_, index) => ({ x: index, y: index, pressure: 0.5, time: index })), 8, 50, false)
    for (let index = 0; index < 50; index += 1) {
      store.getState().commitOperation({ ...operation, id: `stroke-${index}` })
    }
    const retained = store.getState().past.reduce((sum, item) => sum + item.bytes, 0)
    const raw = 50 * (96 + 100 * 48)
    expect(store.getState().past).toHaveLength(50)
    expect(retained).toBeLessThanOrEqual(raw * 1.1)
  })
})

// ─── Action parity ───────────────────────────────────────────────────────────

describe('document store history entries', () => {
  it('undoes a layer rename without bumping the search revision on the way in', () => {
    const store = makeStore()
    const revision = store.getState().document.revision
    store.getState().updateLayer('layer-1', { name: 'Inks' })
    expect(store.getState().document.revision).toBe(revision)
    store.getState().undo()
    expect(store.getState().document.layers[0]!.name).toBe('Layer 1')
    expect(store.getState().document.revision).toBe(revision + 1)
  })

  it('moves layers and restores the exact order', () => {
    const store = makeStore()
    store.getState().moveLayer('layer-1', 1)
    expect(store.getState().document.layers.map((layer) => layer.id)).toEqual(['layer-2', 'layer-1', 'layer-3', 'layer-4'])
    store.getState().moveLayer('layer-1', 1)
    expect(store.getState().document.layers.map((layer) => layer.id)).toEqual(['layer-2', 'layer-3', 'layer-1', 'layer-4'])
    store.getState().undo()
    store.getState().undo()
    expect(store.getState().document.layers.map((layer) => layer.id)).toEqual(['layer-1', 'layer-2', 'layer-3', 'layer-4'])
    expect(store.getState().future).toHaveLength(2)
    store.getState().redo()
    store.getState().redo()
    expect(store.getState().document.layers.map((layer) => layer.id)).toEqual(['layer-2', 'layer-3', 'layer-1', 'layer-4'])
  })

  it('treats out-of-range layer moves as no-ops', () => {
    const store = makeStore()
    const before = store.getState()
    store.getState().moveLayer('layer-1', -1)
    store.getState().moveLayer('layer-4', 1)
    expect(store.getState()).toBe(before)
  })

  it('clears a layer and restores every operation on undo', () => {
    const store = makeStore()
    const first = addStroke(store)
    const second = addStroke(store)
    store.getState().clearLayer('layer-1')
    expect(store.getState().document.layers[0]!.operations).toEqual([])
    store.getState().undo()
    expect(store.getState().document.layers[0]!.operations).toEqual([first, second])
    store.getState().redo()
    expect(store.getState().document.layers[0]!.operations).toEqual([])
  })

  it('undoes and redoes the truncated title', () => {
    const store = makeStore()
    store.getState().setTitle('x'.repeat(120))
    expect(store.getState().document.title).toHaveLength(80)
    store.getState().undo()
    expect(store.getState().document.title).toBe('Untitled drawing')
    store.getState().redo()
    expect(store.getState().document.title).toBe('x'.repeat(80))
  })

  it('treats a same-title call as a no-op', () => {
    const store = makeStore()
    store.getState().setTitle('Same')
    const before = store.getState()
    store.getState().setTitle('Same')
    expect(store.getState()).toBe(before)
    expect(store.getState().past).toHaveLength(1)
  })

  it('undoes and redoes trace changes, and redo does not clobber a resolved image', () => {
    const store = makeStore()
    store.getState().setTrace('asset-1', 'blob:trace')
    expect(store.getState().document.trace).toMatchObject({ assetId: 'asset-1', imageUrl: 'blob:trace', visible: true })
    store.getState().updateTrace({ visible: false, opacity: 0.8, scale: 2 })
    store.getState().setResolvedTraceImage('blob:resolved')
    const historyLength = store.getState().past.length
    store.getState().setResolvedTraceImage('blob:resolved-again')
    expect(store.getState().past).toHaveLength(historyLength)

    store.getState().undo()
    expect(store.getState().document.trace).toEqual({ assetId: 'asset-1', imageUrl: 'blob:resolved-again', visible: true, opacity: 0.3, scale: 1 })
    store.getState().redo()
    expect(store.getState().document.trace).toMatchObject({ visible: false, opacity: 0.8, scale: 2, imageUrl: 'blob:resolved-again' })

    store.getState().clearTrace()
    expect(store.getState().document.trace).toMatchObject({ assetId: null, imageUrl: null, visible: false })
    store.getState().undo()
    expect(store.getState().document.trace).toMatchObject({ assetId: 'asset-1', imageUrl: 'blob:resolved-again' })
  })

  it('resets history on replacement and creation without mutating the input document', () => {
    const store = makeStore()
    addStroke(store)
    const input = structuredClone(store.getState().document)
    const snapshot = structuredClone(input)
    store.getState().replaceDocument(input, 'layer-2')
    expect(store.getState().past).toHaveLength(0)
    expect(store.getState().future).toHaveLength(0)
    expect(store.getState().activeLayerId).toBe('layer-2')
    const before = store.getState()
    store.getState().undo()
    expect(store.getState()).toBe(before)
    expect(input).toEqual(snapshot)
    expect(store.getState().document).not.toBe(input)

    addStroke(store)
    store.getState().newDocument()
    expect(store.getState().past).toHaveLength(0)
    expect(store.getState().future).toHaveLength(0)
    expect(store.getState().document.layers.map((layer) => layer.id)).toEqual(['layer-1', 'layer-2', 'layer-3', 'layer-4'])
    expect(store.getState().activeLayerId).toBe('layer-1')
  })

  it('caps both stacks by entry count', () => {
    const store = makeStore({ maxEntries: 3 })
    for (let index = 0; index < 6; index += 1) addStroke(store)
    expect(store.getState().past).toHaveLength(3)
    store.getState().undo()
    store.getState().undo()
    store.getState().undo()
    expect(store.getState().past).toHaveLength(0)
    expect(store.getState().future).toHaveLength(3)
    store.getState().undo()
    expect(store.getState().future).toHaveLength(3)
  })

  it('evicts oldest entries when the byte budget is exceeded but keeps the newest', () => {
    const single = 96 + 100 * 48
    const store = makeStore({ maxBytes: single * 2 })
    for (let index = 0; index < 5; index += 1) {
      store.getState().commitOperation(makeStrokeOperation('pressure', Array.from({ length: 100 }, (_, point) => ({ x: point, y: point, pressure: 0.5, time: point })), 8, 50, false))
    }
    expect(store.getState().past).toHaveLength(2)

    const tiny = makeStore({ maxBytes: 1 })
    addStroke(tiny)
    expect(tiny.getState().past).toHaveLength(1)
    expect(tiny.getState().past[0]!.bytes).toBeGreaterThan(1)
  })

  it('returns the same state for no-op actions', () => {
    const store = makeStore()
    const before = store.getState()
    store.getState().updateLayer('layer-1', { opacity: 1 })
    store.getState().updateLayer('missing-layer', { opacity: 0.5 })
    store.getState().clearLayer('layer-1')
    store.getState().undo()
    store.getState().redo()
    expect(store.getState()).toBe(before)
  })

  it('clears future on a new edit', () => {
    const store = makeStore()
    addStroke(store)
    store.getState().undo()
    expect(store.getState().future).toHaveLength(1)
    addStroke(store)
    expect(store.getState().future).toHaveLength(0)
  })
})

// ─── Round-trip equivalence ──────────────────────────────────────────────────

describe('document store round trips', () => {
  it('undoes back through every entry kind and redoes forward again', () => {
    const store = makeStore()
    const snapshots: string[] = [JSON.stringify(store.getState().document)]
    const record = () => snapshots.push(JSON.stringify(store.getState().document))

    addStroke(store); record()
    store.getState().updateLayer('layer-1', { name: 'Inks' }); record()
    store.getState().updateLayer('layer-1', { opacity: 0.4, visible: false }); record()
    addStroke(store, 2); record()
    store.getState().moveLayer('layer-1', 1); record()
    store.getState().clearLayer('layer-1'); record()
    store.getState().setTitle('Round trip'); record()
    store.getState().setTrace('asset-1', 'blob:trace'); record()
    store.getState().updateTrace({ scale: 1.5 }); record()
    store.getState().clearTrace(); record()
    addStroke(store); record()
    store.getState().updateLayer('layer-2', { visible: false }); record()

    const normalize = (value: string) => {
      const document = JSON.parse(value) as { revision: number; updatedAt: number }
      document.revision = 0
      document.updatedAt = 0
      return JSON.stringify(document)
    }

    for (let index = snapshots.length - 2; index >= 0; index -= 1) {
      store.getState().undo()
      expect(normalize(JSON.stringify(store.getState().document))).toBe(normalize(snapshots[index]!))
    }
    expect(store.getState().past).toHaveLength(0)

    for (let index = 1; index < snapshots.length; index += 1) {
      store.getState().redo()
      expect(normalize(JSON.stringify(store.getState().document))).toBe(normalize(snapshots[index]!))
    }
    expect(store.getState().future).toHaveLength(0)
  })

  it('stays reversible across an interleaved seeded sequence', () => {
    const store = makeStore()
    let seed = 42
    const random = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648
      return seed / 2147483648
    }

    for (let index = 0; index < 40; index += 1) {
      const roll = random()
      if (roll < 0.4) addStroke(store, 1 + (index % 3))
      else if (roll < 0.55) store.getState().updateLayer(`layer-${1 + (index % 4)}`, { opacity: 0.1 * (index % 10) })
      else if (roll < 0.65) store.getState().moveLayer('layer-2', index % 2 === 0 ? -1 : 1)
      else if (roll < 0.72) store.getState().clearLayer(`layer-${1 + (index % 4)}`)
      else if (roll < 0.8) store.getState().setTitle(`Title ${index}`)
      else if (roll < 0.88) store.getState().updateTrace({ visible: index % 3 !== 0 })
      else if (roll < 0.94) store.getState().undo()
      else store.getState().redo()
    }

    const normalize = () => {
      const document = structuredClone(store.getState().document)
      document.revision = 0
      document.updatedAt = 0
      return JSON.stringify(document)
    }
    const settled = normalize()
    let undos = 0
    while (store.getState().past.length > 0) {
      store.getState().undo()
      undos += 1
      expect(undos).toBeLessThan(2000)
    }
    for (let index = 0; index < undos; index += 1) store.getState().redo()
    expect(normalize()).toBe(settled)
  })
})
