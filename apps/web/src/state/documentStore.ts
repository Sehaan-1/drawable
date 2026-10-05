import { createStore, type StoreApi } from 'zustand/vanilla'
import { useContext, createContext, createElement, type ReactNode } from 'react'
import { useStore } from 'zustand'
import { uid } from '../lib/drawing'
import type { DrawingDocument, StrokeOperation, Tool } from '../lib/types'

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

function cloneDocument(document: DrawingDocument): DrawingDocument {
  return structuredClone(document)
}

export interface DocumentState {
  document: DrawingDocument
  past: DrawingDocument[]
  future: DrawingDocument[]
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

function mutateDocument(
  state: DocumentState,
  mutation: (document: DrawingDocument) => void,
  affectsSearch = true,
): Pick<DocumentState, 'document' | 'past' | 'future'> {
  const next = cloneDocument(state.document)
  mutation(next)
  if (affectsSearch) next.revision += 1
  next.updatedAt = Date.now()
  return {
    document: next,
    past: [...state.past.slice(-499), state.document],
    future: [],
  }
}

export function createDocumentStore(options: DocumentStoreOptions = {}): StoreApi<DocumentState> {
  const emit = options.onCommand ?? (() => undefined)
  return createStore<DocumentState>((set) => ({
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
    setTitle: (title) => set((state) => mutateDocument(state, (document) => { document.title = title.slice(0, 80) }, false)),
    setTool: (activeTool) => set({ activeTool }),
    setBrushSize: (brushSize) => set({ brushSize: Math.min(40, Math.max(1, brushSize)) }),
    setStabilization: (stabilization) => set({ stabilization: Math.min(100, Math.max(0, stabilization)) }),
    setSimulatePressure: (simulatePressure) => set({ simulatePressure }),
    setActiveLayer: (activeLayerId) => set({ activeLayerId }),
    commitOperation: (operation) => set((state) => {
      const layerId = state.activeLayerId
      const layer = state.document.layers.find((candidate) => candidate.id === layerId)
      if (!layer) return state
      emit({ type: 'operation_committed', layerId, operation: structuredClone(operation) })
      return mutateDocument(state, (document) => {
        document.layers.find((candidate) => candidate.id === layerId)?.operations.push(operation)
      })
    }),
    updateLayer: (id, update) => set((state) => {
      const current = state.document.layers.find((layer) => layer.id === id)
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
      return mutateDocument(state, (document) => {
        const layer = document.layers.find((candidate) => candidate.id === id)
        if (!layer) return
        if (update.name !== undefined) layer.name = update.name.slice(0, 40)
        if (update.visible !== undefined) layer.visible = update.visible
        if (update.opacity !== undefined) layer.opacity = Math.min(1, Math.max(0, update.opacity))
      }, update.visible !== undefined || update.opacity !== undefined)
    }),
    moveLayer: (id, direction) => set((state) => {
      const index = state.document.layers.findIndex((layer) => layer.id === id)
      const target = index + direction
      if (index < 0 || target < 0 || target >= state.document.layers.length) return state
      emit({ type: 'layer_moved', layerId: id, direction })
      return mutateDocument(state, (document) => {
        const [layer] = document.layers.splice(index, 1)
        if (layer) document.layers.splice(target, 0, layer)
      })
    }),
    clearLayer: (id) => set((state) => {
      const layer = state.document.layers.find((candidate) => candidate.id === id)
      if (!layer || layer.operations.length === 0) return state
      emit({ type: 'layer_cleared', layerId: id })
      return mutateDocument(state, (document) => {
        const target = document.layers.find((candidate) => candidate.id === id)
        if (target) target.operations = []
      })
    }),
    setTrace: (assetId, imageUrl) => set((state) => {
      emit({ type: 'trace_updated' })
      return mutateDocument(state, (document) => {
        document.trace = { ...document.trace, assetId, imageUrl, visible: true }
      }, false)
    }),
    updateTrace: (update) => set((state) => {
      emit({ type: 'trace_updated' })
      return mutateDocument(state, (document) => {
        document.trace = { ...document.trace, ...update }
      }, false)
    }),
    clearTrace: () => set((state) => {
      emit({ type: 'trace_updated' })
      return mutateDocument(state, (document) => {
        document.trace.assetId = null
        document.trace.imageUrl = null
      }, false)
    }),
    setResolvedTraceImage: (imageUrl) => set((state) => ({ document: { ...state.document, trace: { ...state.document.trace, imageUrl } } })),
    undo: () => set((state) => {
    const previous = state.past.at(-1)
    if (!previous) return state
    emit({ type: 'undo' })
    return { document: { ...cloneDocument(previous), revision: state.document.revision + 1, updatedAt: Date.now() }, past: state.past.slice(0, -1), future: [state.document, ...state.future].slice(0, 500) }
    }),
    redo: () => set((state) => {
    const next = state.future[0]
    if (!next) return state
    emit({ type: 'redo' })
    return { document: { ...cloneDocument(next), revision: state.document.revision + 1, updatedAt: Date.now() }, past: [...state.past, state.document].slice(-500), future: state.future.slice(1) }
    }),
    newDocument: () => set((state) => {
      emit({ type: 'document_created' })
      return { document: { ...makeDocument(), revision: state.document.revision + 1 }, past: [], future: [], activeLayerId: 'layer-1' }
    }),
  }))
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
