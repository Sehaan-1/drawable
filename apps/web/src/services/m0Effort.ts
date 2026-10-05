export const M0_EFFORT_CATEGORIES = [
  'source-review',
  'asset-review',
  'feasibility-drawing',
  'evidence-review',
] as const

export type M0EffortCategory = typeof M0_EFFORT_CATEGORIES[number]

export interface M0EffortChunk {
  schemaVersion: 1
  chunkId: string
  sessionId: string
  category: M0EffortCategory
  startedAt: string
  endedAt: string
  activeMs: number
  checkpoint: 'periodic' | 'paused'
}

const DB_NAME = 'drawable-m0-effort'
const STORE = 'active-effort'

function openDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) {
        request.result.createObjectStore(STORE, { keyPath: 'chunkId' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

function completed(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
    transaction.onabort = () => reject(transaction.error ?? new Error('Effort log write was aborted.'))
  })
}

export async function appendM0EffortChunk(chunk: M0EffortChunk) {
  const db = await openDatabase()
  try {
    const transaction = db.transaction(STORE, 'readwrite')
    const result = completed(transaction)
    transaction.objectStore(STORE).add(structuredClone(chunk))
    await result
  } finally {
    db.close()
  }
}

export async function exportM0EffortLog(): Promise<Blob> {
  const db = await openDatabase()
  try {
    const transaction = db.transaction(STORE, 'readonly')
    const done = completed(transaction)
    const chunks = await new Promise<M0EffortChunk[]>((resolve, reject) => {
      const request = transaction.objectStore(STORE).getAll()
      request.onsuccess = () => resolve(request.result as M0EffortChunk[])
      request.onerror = () => reject(request.error)
    })
    await done
    chunks.sort((left, right) => left.startedAt.localeCompare(right.startedAt) || left.chunkId.localeCompare(right.chunkId))
    return new Blob([JSON.stringify({
      format: 'drawable-m0-effort-log',
      formatVersion: 1,
      exportedAt: new Date().toISOString(),
      chunks,
      active_effort_minutes: Math.floor(chunks.reduce((sum, chunk) => sum + chunk.activeMs, 0) / 60_000),
      accounting: 'Only explicitly running, foreground-visible time is counted. The final uncheckpointed interval after an abrupt browser close is excluded.',
    }, null, 2)], { type: 'application/vnd.drawable.m0-effort+json' })
  } finally {
    db.close()
  }
}
