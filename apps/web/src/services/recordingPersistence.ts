import { checksumRenderedDocument } from './renderChecksum'
import { validateAndReplay, replaySequence } from './recordingValidation'
import { migrateLegacyRecordingV1, migrateLegacyRecordingV2, type DrawingSequenceCurrent, type DrawingSequenceV1, type DrawingSequenceV2Legacy } from './recordingTypes'

const DB_NAME = 'drawable-research-recordings'
const DB_VERSION = 2
const STORE = 'accepted'

function openDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE, { keyPath: 'recordingId' })
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
    request.onblocked = () => reject(new Error('Close other Drawable tabs to finish the local recording-store upgrade.'))
  })
}

function requestValue<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

function transactionComplete(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
    transaction.onabort = () => reject(transaction.error ?? new Error('Recording-store transaction was aborted.'))
  })
}

function beginTransaction(db: IDBDatabase, mode: IDBTransactionMode) {
  try {
    return db.transaction(STORE, mode, { durability: 'strict' })
  } catch {
    return db.transaction(STORE, mode)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function normalizeStoredRecording(value: unknown): DrawingSequenceCurrent {
  if (!isRecord(value) || value.format !== 'drawable-sequence') {
    throw new Error('A stored recording is corrupt or has an unsupported format. The original record was left untouched.')
  }
  if (value.formatVersion === 1) {
    try {
      return migrateLegacyRecordingV1(value as unknown as DrawingSequenceV1).sequence
    } catch {
      throw new Error('A legacy recording could not be migrated safely. Its original record was left untouched.')
    }
  }
  if (value.formatVersion === 2) {
    try {
      return migrateLegacyRecordingV2(value as unknown as DrawingSequenceV2Legacy).sequence
    } catch {
      throw new Error('A legacy version 2 recording could not be migrated safely. Its original record was left untouched.')
    }
  }
  if (value.formatVersion === 3) {
    const sequence = value as unknown as DrawingSequenceCurrent
    const validation = validateAndReplay(sequence)
    if (!validation.valid) throw new Error(`A stored version 3 recording failed validation (${validation.errors.join('; ')}). The original record was left untouched.`)
    return sequence
  }
  throw new Error('A stored recording uses an unsupported version. The original record was left untouched.')
}

export async function saveAcceptedRecording(recording: DrawingSequenceCurrent) {
  if (!recording.acceptedAt || !Number.isFinite(Date.parse(recording.acceptedAt))) {
    throw new Error('Only a recording with a valid acceptance timestamp can be saved as accepted.')
  }
  const validation = validateAndReplay(recording)
  if (!validation.valid) throw new Error(validation.errors.join('; '))
  const replayedChecksum = await checksumRenderedDocument(replaySequence(recording))
  if (replayedChecksum !== recording.finalRenderedSha256) {
    throw new Error('Replay-to-render checksum does not match the accepted recording; nothing was saved.')
  }

  const db = await openDatabase()
  try {
    const transaction = beginTransaction(db, 'readwrite')
    const complete = transactionComplete(transaction)
    transaction.objectStore(STORE).put(structuredClone(recording))
    await complete
  } finally {
    db.close()
  }
}

export async function listAcceptedRecordings(): Promise<DrawingSequenceCurrent[]> {
  const db = await openDatabase()
  try {
    const transaction = db.transaction(STORE, 'readonly')
    const complete = transactionComplete(transaction)
    const raw = await requestValue(transaction.objectStore(STORE).getAll())
    await complete
    return (raw as unknown[]).map(normalizeStoredRecording)
  } finally {
    db.close()
  }
}

/** Build a complete JSON batch in memory before download; no record is deleted or modified. */
export async function exportAcceptedRecordingBatch(): Promise<Blob> {
  const recordings = await listAcceptedRecordings()
  const records: { status: 'needs_review' | 'accepted'; sequence: DrawingSequenceCurrent }[] = []
  for (const sequence of recordings) {
    if (sequence.migrationReview) {
      records.push({ status: 'needs_review', sequence })
      continue
    }
    const validation = validateAndReplay(sequence)
    if (!validation.valid) throw new Error(`Recording ${sequence.recordingId} failed validation; no batch was exported.`)
    const checksum = await checksumRenderedDocument(replaySequence(sequence))
    if (checksum !== sequence.finalRenderedSha256) throw new Error(`Recording ${sequence.recordingId} failed rendered-output verification; no batch was exported.`)
    records.push({ status: 'accepted', sequence })
  }
  const payload = {
    format: 'drawable-sequence-batch',
    formatVersion: 1,
    exportedAt: new Date().toISOString(),
    records,
  }
  return new Blob([JSON.stringify(payload, null, 2)], { type: 'application/vnd.drawable.sequence-batch+json' })
}

export async function deleteAcceptedRecording(recordingId: string) {
  const db = await openDatabase()
  try {
    const transaction = beginTransaction(db, 'readwrite')
    const complete = transactionComplete(transaction)
    transaction.objectStore(STORE).delete(recordingId)
    await complete
  } finally {
    db.close()
  }
}
