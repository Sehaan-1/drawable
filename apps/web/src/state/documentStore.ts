import { createStore, type StoreApi } from 'zustand/vanilla'
import { useContext, createContext, createElement, type ReactNode } from 'react'
import { useStore } from 'zustand'
import { uid } from '../lib/drawing'
import type { DrawingDocument, StrokeOperation, Tool, TraceState } from '../lib/types'
import {
  applyHistoryEntry,
  createHistoryEntry,
  DEFAULT_HISTORY_LIMITS,
  findLayer,
  trimFuture,
  trimPast,
  type HistoryEntry,
  type HistoryLimits,
  type HistoryPayload,
} from './documentHistory'

export type DocumentCommand =
  | { type: 'operation_committed'; layerId: string; operation: StrokeOperation }
  | { type: 'layer_updated'; layerId: string; update: { name?: string; visible?: boolean; opacity?: number } }
  | { type: 'layer_moved'; layerId: string; direction: -1 | 1 }
  | { type: 'layer_cleared'; layerId: string }
  | { type: 'trace_updated' }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'document_replaced' }
  | { type: 'document_created' }

export interface DocumentStoreOptions {
  onCommand?: (command: DocumentCommand) => void
  history?: Partial<HistoryLimits>
}

const makeDocument = (): DrawingDocument => ({
  id: uid('document'),
  title: 'Untitled drawing',
  revision: 0,
  updatedAt: Date.now(),
  layers: Array.from({ length: 4 }, (_, index) => ({
    id: `layer-${index + 1}`,
    name: `Layer ${index + 1}`,
    visible: true,
    opacity: 1,
    operations: [],
  })),
  trace: {
    assetId: null,
    imageUrl: null,
    visible: true,
    opacity: 0.3,
    scale: 1,
  },
})

// Only used at the external ownership boundary (replaceDocument), where the
// caller keeps a reference to the document it handed us. Interactive mutations
// use structural sharing instead and never clone the document.
function cloneDocument(document: DrawingDocument): DrawingDocument {
  return structuredClone(document)
}

export interface DocumentState {
  document: DrawingDocument
  past: HistoryEntry[]
  future: HistoryEntry[]
  activeLayerId: string
  activeTool: Tool
  brushSize: number
  stabilization: number
  simulatePressure: boolean
  hasHydrated: boolean
  setHydrated: (value: boolean) => void
  replaceDocument: (document: DrawingDocument, activeLayerId?: string) => void
  setTitle: (title: string) => void
  setTool: (tool: Tool) => void
  setBrushSize: (size: number) => void
  setStabilization: (value: number) => void
  setSimulatePressure: (value: boolean) => void
  setActiveLayer: (id: string) => void
  commitOperation: (operation: StrokeOperation) => void
  updateLayer: (id: string, update: { name?: string; visible?: boolean; opacity?: number }) => void
  moveLayer: (id: string, direction: -1 | 1) => void
  clearLayer: (id: string) => void
  setTrace: (assetId: string, imageUrl: string) => void
  updateTrace: (update: Partial<Pick<DrawingDocument['trace'], 'visible' | 'opacity' | 'scale'>>) => void
  clearTrace: () => void
  setResolvedTraceImage: (imageUrl: string | null) => void
  undo: () => void
  redo: () => void
  newDocument: () => void
}

export function createDocumentStore(options: DocumentStoreOptions = {}): StoreApi<DocumentState> {
  const emit = options.onCommand ?? (() => undefined)
  const limits: HistoryLimits = { ...DEFAULT_HISTORY_LIMITS, ...options.history }

  return createStore<DocumentState>((set) => {
    /**
     * Applies a reversible entry to the document and records it in `past`.
     * Allocates in proportion to the changed layer, never the whole document.
     */
    function commitEntry(
      state: DocumentState,
      payload: HistoryPayload,
      affectsSearch: boolean,
    ): Pick<DocumentState, 'document' | 'past' | 'future'> {
      const entry = createHistoryEntry(payload)
      const next = applyHistoryEntry(state.document, entry, 'redo')
      return {
        document: {
          ...next,
          revision: state.document.revision + (affectsSearch ? 1 : 0),
          updatedAt: Date.now(),
        },
        past: trimPast([...state.past, entry], limits),
        future: [],
      }
    }

    return {
      document: makeDocument(),
      past: [],
      future: [],
      activeLayerId: 'layer-1',
      activeTool: 'pressure',
      brushSize: 12,
      stabilization: 50,
      simulatePressure: true,
      hasHydrated: false,
      setHydrated: (hasHydrated) => set({ hasHydrated }),
      replaceDocument: (document, activeLayerId) => set((state) => {
        const next = { document: { ...cloneDocument(document), revision: state.document.revision + 1 }, past: [], future: [], activeLayerId: document.layers.some((layer) => layer.id === activeLayerId) ? activeLayerId! : document.layers[0]?.id ?? 'layer-1' }
        emit({ type: 'document_replaced' })
        return next
      }),
      setTitle: (title) => set((state) => {
        const nextTitle = title.slice(0, 80)
        if (nextTitle === state.document.title) return state
        return commitEntry(state, { kind: 'title_changed', before: state.document.title, after: nextTitle }, false)
      }),
      setTool: (activeTool) => set({ activeTool }),
      setBrushSize: (brushSize) => set({ brushSize: Math.min(40, Math.max(1, brushSize)) }),
      setStabilization: (stabilization) => set({ stabilization: Math.min(100, Math.max(0, stabilization)) }),
      setSimulatePressure: (simulatePressure) => set({ simulatePressure }),
      setActiveLayer: (activeLayerId) => set({ activeLayerId }),
      commitOperation: (operation) => set((state) => {
        const layerId = state.activeLayerId
        if (!findLayer(state.document, layerId)) return state
        // Recorder payloads must not alias live store data; only pay for the
        // snapshot when a listener is actually registered.
        if (options.onCommand) {
          options.onCommand({ type: 'operation_committed', layerId, operation: structuredClone(operation) })
        }
        // Shallow copy establishes the store's ownership of the operation object
        // while sharing the (immutable after commit) points array.
        return commitEntry(state, { kind: 'operation_appended', layerId, operation: { ...operation } }, true)
      }),
      updateLayer: (id, update) => set((state) => {
        const current = findLayer(state.document, id)
        if (!current) return state
        const nextName = update.name === undefined ? current.name : update.name.slice(0, 40)
        const nextVisible = update.visible === undefined ? current.visible : update.visible
        const nextOpacity = update.opacity === undefined ? current.opacity : Math.min(1, Math.max(0, update.opacity))
        if (nextName === current.name && nextVisible === current.visible && nextOpacity === current.opacity) return state
        emit({ type: 'layer_updated', layerId: id, update: {
          ...(update.name === undefined ? {} : { name: nextName }),
          ...(update.visible === undefined ? {} : { visible: nextVisible }),
          ...(update.opacity === undefined ? {} : { opacity: nextOpacity }),
        } })
        return commitEntry(state, {
          kind: 'layer_properties_changed',
          layerId: id,
          before: { name: current.name, visible: current.visible, opacity: current.opacity },
          after: { name: nextName, visible: nextVisible, opacity: nextOpacity },
        }, update.visible !== undefined || update.opacity !== undefined)
      }),
      moveLayer: (id, direction) => set((state) => {
        const layers = state.document.layers
        const index = layers.findIndex((layer) => layer.id === id)
        const target = index + direction
        if (index < 0 || target < 0 || target >= layers.length) return state
        emit({ type: 'layer_moved', layerId: id, direction })
        return commitEntry(state, { kind: 'layer_moved', layerId: id, from: index, to: target }, true)
      }),
      clearLayer: (id) => set((state) => {
        const layer = findLayer(state.document, id)
        if (!layer || layer.operations.length === 0) return state
        emit({ type: 'layer_cleared', layerId: id })
        // Retains the existing immutable operations array by reference.
        return commitEntry(state, { kind: 'layer_cleared', layerId: id, operations: layer.operations }, true)
      }),
      setTrace: (assetId, imageUrl) => set((state) => {
        emit({ type: 'trace_updated' })
        const trace = state.document.trace
        return commitEntry(state, {
          kind: 'trace_changed',
          before: { assetId: trace.assetId, imageUrl: trace.imageUrl, visible: trace.visible },
          after: { assetId, imageUrl, visible: true },
        }, false)
      }),
      updateTrace: (update) => set((state) => {
        emit({ type: 'trace_updated' })
        const before: TraceState = state.document.trace
        const after: Partial<Pick<TraceState, 'assetId' | 'imageUrl' | 'visible' | 'opacity' | 'scale'>> = {}
        const previous: typeof after = {}
        for (const key of ['visible', 'opacity', 'scale'] as const) {
          if (update[key] === undefined) continue
          Object.assign(previous, { [key]: before[key] })
          Object.assign(after, { [key]: update[key] })
        }
        return commitEntry(state, { kind: 'trace_changed', before: previous, after }, false)
      }),
      clearTrace: () => set((state) => {
        emit({ type: 'trace_updated' })
        const trace = state.document.trace
        return commitEntry(state, {
          kind: 'trace_changed',
          before: { assetId: trace.assetId, imageUrl: trace.imageUrl },
          after: { assetId: null, imageUrl: null },
        }, false)
      }),
      setResolvedTraceImage: (imageUrl) => set((state) => {
        if (state.document.trace.imageUrl === imageUrl) return state
        return { document: { ...state.document, trace: { ...state.document.trace, imageUrl } } }
      }),
      undo: () => set((state) => {
        const entry = state.past.at(-1)
        if (!entry) return state
        emit({ type: 'undo' })
        const next = applyHistoryEntry(state.document, entry, 'undo')
        return {
          document: { ...next, revision: state.document.revision + 1, updatedAt: Date.now() },
          past: state.past.slice(0, -1),
          future: trimFuture([entry, ...state.future], limits),
        }
      }),
      redo: () => set((state) => {
        const entry = state.future[0]
        if (!entry) return state
        emit({ type: 'redo' })
        const next = applyHistoryEntry(state.document, entry, 'redo')
        return {
          document: { ...next, revision: state.document.revision + 1, updatedAt: Date.now() },
          past: trimPast([...state.past, entry], limits),
          future: state.future.slice(1),
        }
      }),
      newDocument: () => set((state) => {
        emit({ type: 'document_created' })
        return { document: { ...makeDocument(), revision: state.document.revision + 1 }, past: [], future: [], activeLayerId: 'layer-1' }
      }),
    }
  })
}

export const documentStore = createDocumentStore()
const DocumentStoreContext = createContext<StoreApi<DocumentState>>(documentStore)

export function DocumentStoreProvider({ store, children }: { store: StoreApi<DocumentState>; children: ReactNode }) {
  return createElement(DocumentStoreContext.Provider, { value: store }, children)
}

export function useDocumentStoreApi(): StoreApi<DocumentState> {
  return useContext(DocumentStoreContext)
}

export function useDocumentStore<T>(selector: (state: DocumentState) => T): T {
  return useStore(useDocumentStoreApi(), selector)
}

// Imperative compatibility for existing non-React tests and services that use the default draw store.
useDocumentStore.getState = documentStore.getState
useDocumentStore.setState = documentStore.setState
useDocumentStore.subscribe = documentStore.subscribe
