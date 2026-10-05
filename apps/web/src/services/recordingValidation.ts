import { LOGICAL_SIZE, type DrawingDocument, type DrawingOperation, type StrokeOperation } from '../lib/types'
import { createDocumentStore, type DocumentCommand } from '../state/documentStore'
import {
  MAX_RECORDING_EVENTS, MAX_RECORDING_OPERATIONS, MAX_RECORDING_POINTS, RECORDING_FORMAT,
  RECORDING_FORMAT_VERSION, RECORDING_SCOPES, RECORDING_STYLES,
  type DrawingSequenceCurrent, type DrawingSequenceV2Legacy, type RecordingValidationResult,
} from './recordingTypes'

const clone = <T>(value: T): T => structuredClone(value)

function validStroke(operation: DrawingOperation): operation is StrokeOperation {
  return operation && operation.kind === 'stroke' && operation.tool !== undefined && Array.isArray(operation.points) && operation.points.every((point) => (
    Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.pressure) && Number.isFinite(point.time)
      && point.x >= 0 && point.x <= LOGICAL_SIZE && point.y >= 0 && point.y <= LOGICAL_SIZE
      && point.pressure >= 0 && point.pressure <= 1 && point.time >= 0
  ))
}

export function validateRecording(bundle: unknown): RecordingValidationResult {
  const errors: string[] = []
  if (!bundle || typeof bundle !== 'object') return { valid: false, errors: ['Recording must be an object'] }
  const value = bundle as Partial<DrawingSequenceCurrent>
  const formatVersion = (bundle as { formatVersion?: unknown }).formatVersion
  const isCurrentVersion = formatVersion === RECORDING_FORMAT_VERSION
  if (value.format !== RECORDING_FORMAT) errors.push('Unsupported recording format')
  if (formatVersion !== RECORDING_FORMAT_VERSION && formatVersion !== 2) errors.push('Unsupported recording version')
  if (!value.recordingId || !value.contributorGroupId) errors.push('Recording identity is missing')
  if (!value.contributorGroupId || value.contributorGroupId.length > 100) errors.push('Pseudonymous contributor-group code is missing or too long')
  if (!value.task || !value.task.taskId || value.task.taskId.length > 100 || !value.task.instructions || value.task.instructions.length > 1000 || !RECORDING_SCOPES.includes(value.task.intendedScope) || !RECORDING_STYLES.includes(value.task.intendedStyle) || (isCurrentVersion && !['pilot-development', 'full-program'].includes(value.task.collectionProgram))) errors.push('Task instructions, ID, scope, style, or collection program are invalid')
  if (!value.consent?.creatorConfirmed || !value.consent.contributorCanGrantPermissions || !value.consent.referenceRightsDeclared || !value.consent.research || !value.consent.evaluation) errors.push('Required creator, reference-rights, research, and evaluation consent is missing')
  if (value.consent?.cloudProcessing && !value.consent.cloudStorage) errors.push('Cloud processing consent requires separate cloud-storage consent')
  if ((value.consent?.cloudStorage || value.consent?.cloudProcessing) && (!value.consent.research || !value.consent.evaluation)) errors.push('Cloud handling requires research and evaluation consent')
  if (value.reference?.kind === 'external') {
    try {
      const url = new URL(value.reference.sourceUrl ?? '')
      if (url.protocol !== 'https:' || url.username || url.password) errors.push('External references require an HTTPS URL without embedded credentials')
      if ([...url.searchParams.keys()].some((key) => ['token', 'access_token', 'api_key', 'key', 'secret', 'password', 'auth', 'signature', 'sig', 'session'].includes(key.toLowerCase()))) errors.push('External reference URLs may not contain secret query parameters')
    } catch { errors.push('External reference URL is invalid') }
    if (!['owned', 'licensed', 'public-domain'].includes(value.reference.rightsBasis ?? '')) errors.push('External reference rights basis is missing or invalid')
  } else if (value.reference?.kind === 'assigned-gallery') {
    if (!value.reference.assetId || !['owned', 'licensed', 'public-domain'].includes(value.reference.rightsBasis ?? '')) errors.push('Assigned reference provenance or rights basis is missing')
  } else if (value.reference?.kind !== 'none') errors.push('Reference kind is invalid')
  if (value.reference?.kind !== 'none') {
    const permissions = value.reference?.permittedUses
    const permissionKeys = isCurrentVersion
      ? ['localDisplay', 'training', 'derivation', 'traceViewing', 'export', 'publicDemonstration', 'redistribution', 'cloudStorage', 'cloudProcessing'] as const
      : ['localDisplay', 'training', 'derivation', 'traceViewing', 'export', 'publicDemonstration', 'redistribution'] as const
    if (!permissions || permissionKeys.some((key) => typeof permissions[key] !== 'boolean')) errors.push('Reference permitted-use record is incomplete')
    if (permissions?.cloudProcessing && !permissions.cloudStorage) errors.push('Reference cloud processing requires separate cloud-storage permission')
  }
  if (value.labelReview?.status !== 'pending' && value.labelReview?.status !== 'approved') errors.push('Label review status is missing')
  if (!value.finalRenderedSha256 || !/^[0-9a-f]{64}$/.test(value.finalRenderedSha256)) errors.push('Canonical rendered-pixel checksum is missing or malformed')
  if (value.acceptedAt !== null && (!value.acceptedAt || !Number.isFinite(Date.parse(value.acceptedAt)))) errors.push('Acceptance timestamp is invalid')
  if (!value.createdAt || !Number.isFinite(Date.parse(value.createdAt)) || !value.completedAt || !Number.isFinite(Date.parse(value.completedAt))) errors.push('Recording creation or completion timestamp is invalid')
  if (!Number.isFinite(value.elapsedMs) || (value.elapsedMs ?? -1) < 0) errors.push('Elapsed time is invalid')
  if (!value.document || !Array.isArray(value.document.layers) || value.document.layers.length !== 4) errors.push('A recording must contain exactly four layers')
  const layers = Array.isArray(value.document?.layers) ? value.document.layers : []
  const expectedLayerIds = ['layer-1', 'layer-2', 'layer-3', 'layer-4']
  if (layers.length === 4 && expectedLayerIds.some((id) => !layers.some((layer) => layer?.id === id))) errors.push('Recording layers must use the four stable layer IDs')
  const ids = new Set<string>()
  let operations = 0
  let points = 0
  let strokes = 0
  let erasers = 0
  for (const layer of layers) {
    if (!layer || typeof layer !== 'object') { errors.push('Layer records are malformed'); continue }
    if (!layer.id || ids.has(layer.id)) errors.push('Layer IDs must be unique')
    if (layer.id) ids.add(layer.id)
    if (typeof layer.name !== 'string' || typeof layer.visible !== 'boolean' || typeof layer.opacity !== 'number' || !Number.isFinite(layer.opacity) || layer.opacity < 0 || layer.opacity > 1) errors.push('Layer properties are malformed')
    if (!Array.isArray(layer.operations)) { errors.push('Layer operations are malformed'); continue }
    for (const operation of layer.operations) {
      if (!operation || typeof operation !== 'object') { errors.push('Operation records are malformed'); continue }
      operations += 1
      if (!operation.id || ids.has(operation.id)) errors.push('Operation IDs must be unique')
      if (operation.id) ids.add(operation.id)
      if (!validStroke(operation)) errors.push('Only finite vector stroke and eraser operations are allowed')
      if (operation.kind === 'stroke') {
        points += operation.points.length
        strokes += operation.tool === 'eraser' ? 0 : 1
        erasers += operation.tool === 'eraser' ? 1 : 0
      }
    }
  }
  if (operations > MAX_RECORDING_OPERATIONS) errors.push('Operation limit exceeded')
  if (points > MAX_RECORDING_POINTS) errors.push('Point limit exceeded')
  if (!Array.isArray(value.events)) errors.push('Events are malformed')
  if (Array.isArray(value.events) && value.events.length > MAX_RECORDING_EVENTS) errors.push('Event limit exceeded')
  let previousAt = -1
  let previousSeq = 0
  let previousRevision = -1
  const eventIds = new Set<string>()
  for (const event of Array.isArray(value.events) ? value.events : []) {
    if (!event || typeof event !== 'object') { errors.push('Event records are malformed'); continue }
    if (event.seq !== previousSeq + 1) errors.push('Event sequence is not contiguous')
    if (!Number.isFinite(event.atMs) || event.atMs < previousAt) errors.push('Event times must be monotonic')
    if (!event.eventId || eventIds.has(event.eventId)) errors.push('Event IDs must be present and unique')
    if (!Number.isInteger(event.documentRevision) || event.documentRevision < previousRevision) errors.push('Document revisions must be monotonic nonnegative integers')
    if (!['layer-1', 'layer-2', 'layer-3', 'layer-4'].includes(event.activeLayerId)) errors.push('Event active layer is invalid')
    if (!event.action || typeof event.action !== 'object' || !['recording_started', 'recording_finished', 'document_mutation'].includes(event.action.type)) errors.push('Event action is unsupported')
    const command = event.action?.type === 'document_mutation' ? event.action.command : null
    if (event.action?.type === 'document_mutation' && (!command || typeof command !== 'object' || !['operation_committed', 'layer_updated', 'layer_moved', 'layer_cleared', 'undo', 'redo'].includes(command.type))) errors.push('Event command is unsupported')
    if (command && !('layerId' in command) && command.type !== 'undo' && command.type !== 'redo') errors.push('Event layer identity is missing')
    if (command?.type === 'operation_committed' && (!command.layerId || !validStroke(command.operation))) errors.push('Operation event is malformed')
    previousSeq = event.seq
    previousAt = event.atMs
    previousRevision = event.documentRevision
    if (event.eventId) eventIds.add(event.eventId)
  }
  if (previousAt > (value.elapsedMs ?? -1)) errors.push('Event time exceeds recorded elapsed time')
  if (value.events?.length && (value.events[0]?.action.type !== 'recording_started' || value.events.at(-1)?.action.type !== 'recording_finished')) errors.push('Sequence requires explicit start and finish events')
  if (!['layer-1', 'layer-2', 'layer-3', 'layer-4'].includes(value.activeLayerId ?? '')) errors.push('Final active layer is invalid')
  if (strokes === 0) errors.push('At least one visible stroke is required')
  if (value.document?.trace?.assetId || value.document?.trace?.imageUrl) errors.push('Trace content is not permitted in a recording')
  return { valid: errors.length === 0, errors: [...new Set(errors)] }
}

interface ReplayState {
  document: DrawingDocument
  past: DrawingDocument[]
  future: DrawingDocument[]
}

function applyCommand(state: ReplayState, command: DocumentCommand): ReplayState {
  const next = clone(state.document)
  const history = [...state.past, state.document].slice(-500)
  if (command.type === 'operation_committed') {
    next.layers.find((layer) => layer.id === command.layerId)?.operations.push(clone(command.operation))
    return { document: next, past: history, future: [] }
  }
  if (command.type === 'layer_updated') {
    const layer = next.layers.find((candidate) => candidate.id === command.layerId)
    if (layer) {
      if (command.update.name !== undefined) layer.name = command.update.name.slice(0, 40)
      if (command.update.visible !== undefined) layer.visible = command.update.visible
      if (command.update.opacity !== undefined) layer.opacity = command.update.opacity
    }
    return { document: next, past: history, future: [] }
  }
  if (command.type === 'layer_moved') {
    const index = next.layers.findIndex((layer) => layer.id === command.layerId)
    const target = index + command.direction
    if (index >= 0 && target >= 0 && target < next.layers.length) {
      const [layer] = next.layers.splice(index, 1)
      if (layer) next.layers.splice(target, 0, layer)
    }
    return { document: next, past: history, future: [] }
  }
  if (command.type === 'layer_cleared') {
    const layer = next.layers.find((candidate) => candidate.id === command.layerId)
    if (layer) layer.operations = []
    return { document: next, past: history, future: [] }
  }
  if (command.type === 'undo') {
    const previous = state.past.at(-1)
    return previous ? { document: clone(previous), past: state.past.slice(0, -1), future: [state.document, ...state.future].slice(0, 500) } : state
  }
  if (command.type === 'redo') {
    const future = state.future[0]
    return future ? { document: clone(future), past: [...state.past, state.document].slice(-500), future: state.future.slice(1) } : state
  }
  return state
}

export function replaySequence(bundle: DrawingSequenceV2Legacy | DrawingSequenceCurrent): DrawingDocument {
  const initial = createDocumentStore().getState().document
  initial.id = bundle.document.id
  initial.title = bundle.document.title
  let state: ReplayState = { document: initial, past: [], future: [] }
  for (const event of bundle.events) {
    if (event.action.type === 'document_mutation') state = applyCommand(state, event.action.command)
  }
  return state.document
}

export function documentsEquivalent(left: DrawingDocument, right: DrawingDocument): boolean {
  const normalize = (document: DrawingDocument) => {
    const copy = clone(document)
    copy.revision = 0
    copy.updatedAt = 0
    return copy
  }
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right))
}

export function validateAndReplay(bundle: DrawingSequenceV2Legacy | DrawingSequenceCurrent): RecordingValidationResult {
  const result = validateRecording(bundle)
  if (!result.valid) return result
  const replayed = replaySequence(bundle)
  if (!documentsEquivalent(replayed, bundle.document)) return { valid: false, errors: ['Replay does not match the recorded final document'] }
  return result
}

export function countDocument(document: DrawingDocument) {
  const operations = document.layers.flatMap((layer) => layer.operations)
  return {
    operationCount: operations.length,
    pointCount: operations.reduce((sum, operation) => sum + (operation.kind === 'stroke' ? operation.points.length : 0), 0),
    strokeCount: operations.filter((operation) => operation.kind === 'stroke' && operation.tool !== 'eraser').length,
    eraserCount: operations.filter((operation) => operation.kind === 'stroke' && operation.tool === 'eraser').length,
  }
}
