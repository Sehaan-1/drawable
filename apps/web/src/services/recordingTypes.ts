import type { DrawingDocument, StrokeOperation } from '../lib/types'
import type { DocumentCommand } from '../state/documentStore'

export const RECORDING_FORMAT = 'drawable-sequence' as const
export const RECORDING_FORMAT_VERSION = 3 as const
export const RECORDING_MIME = 'application/vnd.drawable.sequence+json'
export const RECORDING_EXTENSION = '.drawable-sequence.json'
export const MAX_RECORDING_OPERATIONS = 10_000
export const MAX_RECORDING_POINTS = 1_000_000
export const MAX_RECORDING_EVENTS = 20_000

export const RECORDING_SCOPES = [
  'eye', 'eyebrow', 'mouth', 'face_head', 'hair', 'hand', 'foot',
  'upper_body_clothing', 'full_body', 'multi_character', 'unknown',
] as const
export type RecordingScope = typeof RECORDING_SCOPES[number]

export const RECORDING_STYLES = [
  'manga_anime', 'western_ink', 'realistic_academic', 'cartoon', 'gesture_sketch',
] as const
export type RecordingStyle = typeof RECORDING_STYLES[number]

export type RecordingRightsBasis = 'owned' | 'licensed' | 'public-domain'
export type RecordingReferenceKind = 'none' | 'external' | 'assigned-gallery'
export interface RecordingReferencePermissionsV2 {
  localDisplay: boolean
  training: boolean
  derivation: boolean
  traceViewing: boolean
  export: boolean
  publicDemonstration: boolean
  redistribution: boolean
}
export interface RecordingReferencePermissions extends RecordingReferencePermissionsV2 {
  cloudStorage: boolean
  cloudProcessing: boolean
}
export type RecordingMutation = Extract<
  DocumentCommand,
  { type: 'operation_committed' | 'layer_updated' | 'layer_moved' | 'layer_cleared' | 'undo' | 'redo' }
>

export interface RecordingConsentV2 {
  consentVersion: string
  acceptedAt: string
  creatorConfirmed: boolean
  contributorCanGrantPermissions: boolean
  referenceRightsDeclared: boolean
  research: boolean
  training: boolean
  evaluation: boolean
  publicDemo: boolean
  cloudStorage: boolean
  cloudProcessing: boolean
  retention: 'project-retained-until-withdrawal'
}

export interface RecordingTaskV2 {
  taskId: string
  instructions: string
  intendedScope: RecordingScope
  intendedStyle: RecordingStyle
  split: 'pilot_development'
}

export interface RecordingReferenceV2 {
  kind: RecordingReferenceKind
  assetId?: string
  sourceUrl?: string
  title?: string
  rightsBasis?: RecordingRightsBasis
  licenseId?: string
  attribution?: string
  /** Explicitly recorded source-use scope; these are not implied by academic intent. */
  permittedUses?: RecordingReferencePermissionsV2
}

export interface RecordingReferenceV3 extends Omit<RecordingReferenceV2, 'permittedUses'> {
  permittedUses?: RecordingReferencePermissions
}

export type RecordingEventActionV2 =
  | { type: 'recording_started' }
  | { type: 'document_mutation'; command: RecordingMutation }
  | { type: 'recording_finished' }

export interface RecordingEventV2 {
  eventId: string
  seq: number
  atMs: number
  documentRevision: number
  activeLayerId: 'layer-1' | 'layer-2' | 'layer-3' | 'layer-4'
  action: RecordingEventActionV2
}

export interface DrawingSequenceV2Legacy {
  format: typeof RECORDING_FORMAT
  formatVersion: 2
  applicationVersion: string
  rendererVersion: string
  recordingId: string
  contributorGroupId: string
  createdAt: string
  completedAt: string
  acceptedAt: string | null
  elapsedMs: number
  task: RecordingTaskV2
  consent: RecordingConsentV2
  reference: RecordingReferenceV2
  activeLayerId: 'layer-1' | 'layer-2' | 'layer-3' | 'layer-4'
  events: RecordingEventV2[]
  document: DrawingDocument
  finalRenderedSha256: string
  labelReview: { status: 'pending' | 'approved'; note?: string }
  migrationReview?: { sourceVersion: number; status: 'pending'; notes: string[]; legacyLabels?: Record<string, string> }
  statistics: {
    operationCount: number
    pointCount: number
    strokeCount: number
    eraserCount: number
  }
}

export interface DrawingSequenceV3 extends Omit<DrawingSequenceV2Legacy, 'formatVersion' | 'reference'> {
  formatVersion: typeof RECORDING_FORMAT_VERSION
  reference: RecordingReferenceV3
  task: RecordingTaskV2 & { collectionProgram: 'pilot-development' | 'full-program' }
}

/** Version 2 is a readable legacy schema, never an upload/training authorization. */
export type DrawingSequenceV2 = DrawingSequenceV2Legacy
export type DrawingSequenceCurrent = DrawingSequenceV3

/** Version 1 remains readable for an explicit, review-required migration. */
export const LEGACY_RECORDING_SCOPES_V1 = [
  'portrait', 'full-body', 'gesture', 'hands', 'face', 'figure-in-environment',
  'character-turnaround', 'costume', 'action-pose', 'expression', 'unknown',
] as const
export const LEGACY_RECORDING_STYLES_V1 = ['manga-anime', 'western-ink', 'realistic', 'cartoon', 'gesture'] as const
export type LegacyRecordingScopeV1 = typeof LEGACY_RECORDING_SCOPES_V1[number]
export type LegacyRecordingStyleV1 = typeof LEGACY_RECORDING_STYLES_V1[number]

export interface RecordingConsentV1 {
  consentVersion: string
  acceptedAt: string
  creatorConfirmed: boolean
  contributorCanGrantPermissions: boolean
  referenceRightsDeclared: boolean
  research: boolean
  training: boolean
  evaluation: boolean
  publicDemo: boolean
  retention: 'project-retained-until-withdrawal'
}

export interface RecordingEventV1 {
  seq: number
  atMs: number
  command: RecordingMutation
}

export interface DrawingSequenceV1 {
  format: typeof RECORDING_FORMAT
  formatVersion: 1
  applicationVersion: string
  rendererVersion: string
  recordingId: string
  contributorGroupId: string
  createdAt: string
  acceptedAt: string
  elapsedMs: number
  task: { taskId: string; instructions: string; intendedScope: LegacyRecordingScopeV1; intendedStyle: LegacyRecordingStyleV1; split: 'pilot_development' }
  consent: RecordingConsentV1
  reference: RecordingReferenceV2
  events: RecordingEventV1[]
  document: DrawingDocument
  statistics: DrawingSequenceV2['statistics']
}

export interface RecordingValidationResult {
  valid: boolean
  errors: string[]
}

export interface LegacyMigrationResult {
  status: 'needs_review'
  sequence: DrawingSequenceV3
}

export type StoredDrawingSequenceV2 = DrawingSequenceV2Legacy

/** @deprecated Historical code used this name for the current type. */
export type DrawingSequenceV2Current = DrawingSequenceV3

const LEGACY_STYLE_MAP: Record<LegacyRecordingStyleV1, RecordingStyle> = {
  'manga-anime': 'manga_anime',
  'western-ink': 'western_ink',
  realistic: 'realistic_academic',
  cartoon: 'cartoon',
  gesture: 'gesture_sketch',
}

const LEGACY_SCOPE_MAP: Record<LegacyRecordingScopeV1, RecordingScope> = {
  portrait: 'unknown',
  'full-body': 'full_body',
  gesture: 'unknown',
  hands: 'hand',
  face: 'face_head',
  'figure-in-environment': 'unknown',
  'character-turnaround': 'unknown',
  costume: 'upper_body_clothing',
  'action-pose': 'unknown',
  expression: 'unknown',
  unknown: 'unknown',
}

export function migrateLegacyRecordingV1(value: DrawingSequenceV1): LegacyMigrationResult {
  const notes = [
    'Cloud storage and processing permissions were not collected in version 1 and default to false.',
    'Legacy intended scope/style labels require a reviewer to confirm the version 3 taxonomy.',
    'Version 1 did not store a canonical rendered-pixel checksum or explicit completion event.',
  ]
  const scope = LEGACY_SCOPE_MAP[value.task.intendedScope]
  if (scope === 'unknown' && value.task.intendedScope !== 'unknown') {
    notes.push(`Legacy scope '${value.task.intendedScope}' was retained for review and mapped to unknown.`)
  }
  const events: RecordingEventV2[] = value.events.map((event, index) => ({
    eventId: `${value.recordingId}:v1-event:${index + 1}`,
    seq: index + 1,
    atMs: event.atMs,
    documentRevision: 0,
    activeLayerId: 'layer-1',
    action: { type: 'document_mutation', command: structuredClone(event.command) },
  }))
  const reference: RecordingReferenceV3 = value.reference.kind === 'none'
    ? { kind: 'none' }
    : {
        kind: value.reference.kind,
        ...(value.reference.assetId === undefined ? {} : { assetId: value.reference.assetId }),
        ...(value.reference.sourceUrl === undefined ? {} : { sourceUrl: value.reference.sourceUrl }),
        ...(value.reference.title === undefined ? {} : { title: value.reference.title }),
        ...(value.reference.rightsBasis === undefined ? {} : { rightsBasis: value.reference.rightsBasis }),
        ...(value.reference.licenseId === undefined ? {} : { licenseId: value.reference.licenseId }),
        ...(value.reference.attribution === undefined ? {} : { attribution: value.reference.attribution }),
        permittedUses: {
          localDisplay: value.reference.permittedUses?.localDisplay ?? false,
          training: value.reference.permittedUses?.training ?? false,
          derivation: value.reference.permittedUses?.derivation ?? false,
          traceViewing: value.reference.permittedUses?.traceViewing ?? false,
          export: value.reference.permittedUses?.export ?? false,
          publicDemonstration: value.reference.permittedUses?.publicDemonstration ?? false,
          redistribution: value.reference.permittedUses?.redistribution ?? false,
          cloudStorage: false,
          cloudProcessing: false,
        },
      }
  return {
    status: 'needs_review',
    sequence: {
      format: RECORDING_FORMAT,
      formatVersion: RECORDING_FORMAT_VERSION,
      applicationVersion: value.applicationVersion,
      rendererVersion: value.rendererVersion,
      recordingId: value.recordingId,
      contributorGroupId: value.contributorGroupId,
      createdAt: value.createdAt,
      completedAt: value.acceptedAt,
      acceptedAt: value.acceptedAt,
      elapsedMs: value.elapsedMs,
      task: {
        ...structuredClone(value.task),
        intendedScope: scope,
        intendedStyle: LEGACY_STYLE_MAP[value.task.intendedStyle],
        collectionProgram: 'pilot-development',
      },
      consent: { ...structuredClone(value.consent), cloudStorage: false, cloudProcessing: false },
      reference,
      activeLayerId: 'layer-1',
      events,
      document: structuredClone(value.document),
      finalRenderedSha256: '',
      labelReview: { status: 'pending' },
      migrationReview: {
        sourceVersion: 1,
        status: 'pending',
        notes,
        legacyLabels: { intendedScope: value.task.intendedScope, intendedStyle: value.task.intendedStyle },
      },
      statistics: structuredClone(value.statistics),
    },
  }
}

/** Migrate V2 without inventing reference-level cloud permissions. */
export function migrateLegacyRecordingV2(value: DrawingSequenceV2Legacy): LegacyMigrationResult {
  const notes = [
    'Version 2 did not record cloud-storage or cloud-processing permission for referenced artwork; both default to false.',
    'Version 2 records remain review-required before any training or cloud package is accepted.',
  ]
  const reference: RecordingReferenceV3 = value.reference.kind === 'none'
    ? { kind: 'none' }
    : {
        kind: value.reference.kind,
        ...(value.reference.assetId === undefined ? {} : { assetId: value.reference.assetId }),
        ...(value.reference.sourceUrl === undefined ? {} : { sourceUrl: value.reference.sourceUrl }),
        ...(value.reference.title === undefined ? {} : { title: value.reference.title }),
        ...(value.reference.rightsBasis === undefined ? {} : { rightsBasis: value.reference.rightsBasis }),
        ...(value.reference.licenseId === undefined ? {} : { licenseId: value.reference.licenseId }),
        ...(value.reference.attribution === undefined ? {} : { attribution: value.reference.attribution }),
        permittedUses: {
          localDisplay: value.reference.permittedUses?.localDisplay ?? false,
          training: value.reference.permittedUses?.training ?? false,
          derivation: value.reference.permittedUses?.derivation ?? false,
          traceViewing: value.reference.permittedUses?.traceViewing ?? false,
          export: value.reference.permittedUses?.export ?? false,
          publicDemonstration: value.reference.permittedUses?.publicDemonstration ?? false,
          redistribution: value.reference.permittedUses?.redistribution ?? false,
          cloudStorage: false,
          cloudProcessing: false,
        },
      }
  return {
    status: 'needs_review',
    sequence: {
      ...structuredClone(value),
      formatVersion: RECORDING_FORMAT_VERSION,
      task: { ...structuredClone(value.task), collectionProgram: 'pilot-development' },
      reference,
      migrationReview: { sourceVersion: 2, status: 'pending', notes },
    },
  }
}

export function cloneStroke(operation: StrokeOperation): StrokeOperation {
  return structuredClone(operation)
}
