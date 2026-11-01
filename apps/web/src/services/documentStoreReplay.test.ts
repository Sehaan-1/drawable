import { describe, expect, it } from 'vitest'
import { makeStrokeOperation } from '../lib/drawing'
import { createDocumentStore, type DocumentCommand } from '../state/documentStore'
import { documentsEquivalent, replaySequence } from './recordingValidation'

type ReplayBundle = Parameters<typeof replaySequence>[0]

/**
 * The store's history representation is reversible entries; the recording
 * validator replays commands against its own snapshot stack. Both must produce
 * the same document from the same command sequence.
 */
describe('store history vs recording replay', () => {
  it('replays a captured command sequence to the same document', () => {
    const commands: DocumentCommand[] = []
    const store = createDocumentStore({ onCommand: (command) => commands.push(structuredClone(command)) })
    const initialDocument = structuredClone(store.getState().document)

    const stroke = makeStrokeOperation('pressure', [
      { x: 10, y: 10, pressure: 0.5, time: 0 },
      { x: 40, y: 60, pressure: 0.5, time: 16 },
    ], 12, 50, false)

    store.getState().commitOperation(stroke)
    store.getState().updateLayer('layer-2', { opacity: 0.5 })
    store.getState().moveLayer('layer-2', -1)
    store.getState().clearLayer('layer-1')
    store.getState().undo()
    store.getState().redo()
    store.getState().commitOperation({ ...stroke, id: 'stroke-second' })

    const bundle = {
      document: initialDocument,
      events: commands.map((command, index) => ({
        eventId: `event-${index}`,
        seq: index + 1,
        atMs: index,
        documentRevision: index,
        activeLayerId: 'layer-1',
        action: { type: 'document_mutation', command },
      })),
    } as unknown as ReplayBundle

    const replayed = replaySequence(bundle)
    expect(documentsEquivalent(store.getState().document, replayed)).toBe(true)
  })
})
