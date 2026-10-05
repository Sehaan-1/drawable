import { describe, expect, it } from 'vitest'
import { makeStrokeOperation } from '../lib/drawing'
import { createDocumentStore, type DocumentCommand } from '../state/documentStore'
import { countDocument, validateAndReplay, validateRecording } from './recordingValidation'
import { migrateLegacyRecordingV1, migrateLegacyRecordingV2, type DrawingSequenceV1, type DrawingSequenceV2, type RecordingEventV2 } from './recordingTypes'

function makeBundle(): DrawingSequenceV2 {
  let command: DocumentCommand | null = null
  const store = createDocumentStore({ onCommand: (next) => { command = structuredClone(next) } })
  const operation = makeStrokeOperation('pressure', [{ x: 10, y: 20, pressure: 0.5, time: 0 }], 12, 50, true)
  store.getState().commitOperation(operation)
  const document = structuredClone(store.getState().document)
  const recordedCommand = command as unknown as Extract<DocumentCommand, { type: 'operation_committed' }>
  const recorded = structuredClone(recordedCommand.operation)
  recorded.points[0]!.time = 0
  document.layers[0]!.operations[0] = recorded
  const events: RecordingEventV2[] = [
    { eventId: crypto.randomUUID(), seq: 1, atMs: 0, documentRevision: 0, activeLayerId: 'layer-1', action: { type: 'recording_started' } },
    { eventId: crypto.randomUUID(), seq: 2, atMs: 1, documentRevision: 1, activeLayerId: 'layer-1', action: { type: 'document_mutation', command: { ...recordedCommand, operation: recorded } } },
    { eventId: crypto.randomUUID(), seq: 3, atMs: 2, documentRevision: 1, activeLayerId: 'layer-1', action: { type: 'recording_finished' } },
  ]
  const now = new Date(0).toISOString()
  return {
    format: 'drawable-sequence', formatVersion: 2, applicationVersion: 'test', rendererVersion: 'test',
    recordingId: 'recording-test', contributorGroupId: 'participant-test', createdAt: now, completedAt: now, acceptedAt: now, elapsedMs: 2,
    task: { taskId: 'test', instructions: 'test', intendedScope: 'eye', intendedStyle: 'manga_anime', split: 'pilot_development' },
    consent: { consentVersion: 'test', acceptedAt: now, creatorConfirmed: true, contributorCanGrantPermissions: true, referenceRightsDeclared: true, research: true, training: false, evaluation: true, publicDemo: false, cloudStorage: false, cloudProcessing: false, retention: 'project-retained-until-withdrawal' },
    reference: { kind: 'none' }, activeLayerId: 'layer-1', events, document, finalRenderedSha256: 'a'.repeat(64), labelReview: { status: 'pending' }, statistics: countDocument(document),
  }
}

function makeLegacyBundleV1(): DrawingSequenceV1 {
  const current = makeBundle()
  const mutation = current.events[1]!
  if (mutation.action.type !== 'document_mutation') throw new Error('Expected a document mutation')
  return {
    format: 'drawable-sequence', formatVersion: 1, applicationVersion: current.applicationVersion, rendererVersion: current.rendererVersion,
    recordingId: current.recordingId, contributorGroupId: current.contributorGroupId, createdAt: current.createdAt, acceptedAt: current.acceptedAt!, elapsedMs: current.elapsedMs,
    task: { ...current.task, intendedScope: 'portrait', intendedStyle: 'manga-anime', split: 'pilot_development' },
    consent: { ...current.consent, cloudStorage: undefined, cloudProcessing: undefined } as unknown as DrawingSequenceV1['consent'],
    reference: { kind: 'none' }, events: [{ seq: 1, atMs: 1, command: mutation.action.command }], document: current.document, statistics: current.statistics,
  }
}

describe('recording validation and migration', () => {
  it('replays a valid version 2 vector sequence without mutating its source', () => {
    const bundle = makeBundle()
    expect(validateAndReplay(bundle)).toEqual({ valid: true, errors: [] })
  })

  it('requires separate cloud-storage consent, complete provenance, and a render checksum', () => {
    const bundle = makeBundle()
    bundle.consent.cloudProcessing = true
    expect(validateRecording(bundle).errors).toContain('Cloud processing consent requires separate cloud-storage consent')
    const noChecksum = makeBundle()
    noChecksum.finalRenderedSha256 = ''
    expect(validateRecording(noChecksum).valid).toBe(false)
    const external = makeBundle()
    external.reference = { kind: 'external', sourceUrl: 'http://example.org/art.png', rightsBasis: 'licensed' }
    expect(validateRecording(external).valid).toBe(false)
  })

  it('migrates legacy labels into review instead of granting cloud consent or guessing scope', () => {
    const migrated = migrateLegacyRecordingV1(makeLegacyBundleV1())
    expect(migrated.status).toBe('needs_review')
    expect(migrated.sequence.task.intendedScope).toBe('unknown')
    expect(migrated.sequence.migrationReview?.legacyLabels?.intendedScope).toBe('portrait')
    expect(migrated.sequence.consent.cloudStorage).toBe(false)
    expect(migrated.sequence.consent.cloudProcessing).toBe(false)
    expect(migrated.sequence.labelReview.status).toBe('pending')
    expect(migrated.sequence.finalRenderedSha256).toBe('')
  })

  it('migrates V1 and V2 reference permissions explicitly and defaults absent permissions to false', () => {
    const legacyV1 = makeLegacyBundleV1()
    legacyV1.reference = { kind: 'external', title: 'legacy reference' }
    const migratedV1 = migrateLegacyRecordingV1(legacyV1).sequence
    expect(migratedV1.reference.kind).toBe('external')
    if (migratedV1.reference.kind !== 'none') {
      expect(migratedV1.reference.permittedUses).toEqual({
        localDisplay: false, training: false, derivation: false, traceViewing: false,
        export: false, publicDemonstration: false, redistribution: false,
        cloudStorage: false, cloudProcessing: false,
      })
    }

    const legacyV2 = makeBundle()
    legacyV2.reference = { kind: 'assigned-gallery', assetId: 'asset-test' }
    const migratedV2 = migrateLegacyRecordingV2(legacyV2).sequence
    expect(migratedV2.reference.kind).toBe('assigned-gallery')
    if (migratedV2.reference.kind !== 'none') {
      expect(migratedV2.reference.permittedUses).toEqual({
        localDisplay: false, training: false, derivation: false, traceViewing: false,
        export: false, publicDemonstration: false, redistribution: false,
        cloudStorage: false, cloudProcessing: false,
      })
    }
  })

  it('rejects trace content and recordings without a visible stroke', () => {
    const bundle = makeBundle()
    bundle.document.trace.imageUrl = 'data:image/png;base64,invalid'
    expect(validateAndReplay(bundle).valid).toBe(false)
    const noStroke = makeBundle()
    noStroke.document.layers[0]!.operations = []
    noStroke.events = noStroke.events.filter((event) => event.action.type !== 'document_mutation')
    noStroke.statistics = countDocument(noStroke.document)
    expect(validateAndReplay(noStroke).valid).toBe(false)
  })

  it('fails malformed input as a validation result instead of throwing', () => {
    expect(() => validateAndReplay({} as DrawingSequenceV2)).not.toThrow()
    expect(validateAndReplay({ document: { layers: [{ operations: [null] }] } } as unknown as DrawingSequenceV2).valid).toBe(false)
  })
})
