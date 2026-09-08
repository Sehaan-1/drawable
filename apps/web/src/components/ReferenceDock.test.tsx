import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render as renderRaw, screen, waitFor } from '@testing-library/react'
import { TooltipProvider } from '@radix-ui/react-tooltip'
import type { EventRequest } from '@drawable/contracts'
import { ReferenceDock } from './ReferenceDock'
import { useSearchStore } from '../state/searchStore'
import { useDocumentStore } from '../state/documentStore'
import { useUiStore } from '../state/uiStore'
import { useServiceStore } from '../services/serviceRegistry'
import { fixtureServices, localPinClient, type PinClient } from '../services/frontendServices'
import type { ReferenceAsset, SearchResponse } from '../lib/types'

/**
 * The dock is the place where permission and learning meet the artist, so the
 * rules it must not break are: tracing follows the source's permission (never
 * the native/extracted origin), and an interaction event is logged only after
 * the durable state change it describes actually happened.
 */

function asset(id: string, over: Partial<ReferenceAsset> = {}): ReferenceAsset {
  return {
    id,
    title: id,
    imageUrl: `data:image/svg+xml,${id}`,
    style: 'Cartoon',
    scope: 'Eye',
    source: 'Test',
    native: true,
    match: 'Strong',
    traceAllowed: true,
    traceUrl: `data:image/svg+xml,${id}`,
    ...over,
  }
}

/** Native line art the source forbids tracing. */
const nativeForbidden = asset('native-forbidden', { native: true, traceAllowed: false, traceUrl: null })
/** Extracted line art the source explicitly permits tracing. */
const extractedPermitted = asset('extracted-permitted', { native: false, traceAllowed: true })

function response(results: ReferenceAsset[]): SearchResponse {
  return {
    revision: 7,
    generation: 1,
    mode: 'confident',
    interpretation: 'Eye',
    groups: [{ id: 'best', title: 'Best match', results }],
    warning: null,
    countsApproximate: false,
    strokeStatus: 'present',
  }
}

const render = () => renderRaw(<TooltipProvider delayDuration={500}><ReferenceDock /></TooltipProvider>)

let recorded: EventRequest[] = []

function setup(pins: PinClient = localPinClient()) {
  recorded = []
  useServiceStore.setState({
    mode: 'fixture',
    health: { mode: 'fixture', ready: true, message: 'fixture' },
    sessionId: '3fa85f64-5717-4562-b3fc-2c963f66afa6',
    services: {
      ...fixtureServices,
      pins,
      events: { record: async (event) => { recorded.push(event) } },
    },
  })
  useSearchStore.setState({
    gallery: 'fixture',
    pinClient: pins,
    pinned: [],
    revokedPins: [],
    pinsHydrated: true,
    pinError: null,
    selectedAsset: null,
    error: null,
    loading: false,
  })
}

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({ dockMode: 'references', dockCollapsed: false })
  useDocumentStore.getState().newDocument()
  setup()
})

describe('trace permission in the dock', () => {
  it('offers tracing for an extracted asset the source permits', () => {
    useSearchStore.setState({ response: response([extractedPermitted]) })
    render()
    const button = screen.getByRole('button', { name: 'Place on trace layer' })
    expect(button).toBeEnabled()
    fireEvent.click(button)
    expect(useDocumentStore.getState().document.trace.assetId).toBe('extracted-permitted')
  })

  it('refuses tracing for a native asset the source forbids', () => {
    useSearchStore.setState({ response: response([nativeForbidden]) })
    render()
    const button = screen.getByRole('button', { name: 'Tracing is not permitted for this reference' })
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(useDocumentStore.getState().document.trace.assetId).toBeNull()
    expect(recorded).toEqual([])
  })

  it('states the permission in the detail panel and disables its Trace button', () => {
    useSearchStore.setState({ response: response([nativeForbidden]), selectedAsset: nativeForbidden })
    render()
    expect(screen.getByText('Not permitted by the source')).toBeInTheDocument()
    // The origin is still disclosed, but it is not what decides the action.
    expect(screen.getByText('Native line art')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Trace$/ })).toBeDisabled()
  })
})

describe('interaction events', () => {
  it('sends a unique event_uuid and no client style', async () => {
    useSearchStore.setState({ response: response([extractedPermitted]) })
    render()
    fireEvent.click(screen.getByRole('button', { name: 'View extracted-permitted' }))
    fireEvent.click(screen.getAllByRole('button', { name: 'Place on trace layer' })[0]!)
    await waitFor(() => expect(recorded).toHaveLength(2))
    expect(recorded.map((event) => event.event)).toEqual(['open', 'trace'])
    for (const event of recorded) {
      expect(event.event_uuid).toMatch(/^[0-9a-f-]{36}$/)
      expect(event.query_revision).toBe(7)
      expect('style' in event).toBe(false)
    }
    expect(recorded[0]!.event_uuid).not.toBe(recorded[1]!.event_uuid)
  })

  it('logs pin only after the durable pin succeeded, and unpin on the way back', async () => {
    useSearchStore.setState({ response: response([extractedPermitted]) })
    render()
    fireEvent.click(screen.getByRole('button', { name: 'Pin reference' }))
    await waitFor(() => expect(recorded.map((event) => event.event)).toEqual(['pin']))
    expect(useSearchStore.getState().pinned.map((pin) => pin.id)).toEqual(['extracted-permitted'])

    // The card now appears twice (results + the Pinned section); either
    // control unpins the same durable asset.
    expect(screen.getAllByRole('button', { name: 'Unpin reference' })).toHaveLength(2)
    fireEvent.click(screen.getAllByRole('button', { name: 'Unpin reference' })[0]!)
    await waitFor(() => expect(recorded.map((event) => event.event)).toEqual(['pin', 'unpin']))
    expect(useSearchStore.getState().pinned).toEqual([])
  })

  it('logs nothing when the pin never became durable', async () => {
    const failing: PinClient = {
      list: async () => ({ pins: [], revoked: [] }),
      pin: async () => { throw new Error('pin store unavailable') },
      unpin: async () => ({ pins: [], revoked: [] }),
    }
    setup(failing)
    useSearchStore.setState({ response: response([extractedPermitted]) })
    render()
    fireEvent.click(screen.getByRole('button', { name: 'Pin reference' }))
    await screen.findByTestId('pin-error')
    // The state never changed, so the only honest event is "still unpinned".
    expect(recorded.map((event) => event.event)).toEqual(['unpin'])
    expect(useSearchStore.getState().pinned).toEqual([])
  })

  it('never breaks the UI when event logging fails', async () => {
    useServiceStore.setState({
      services: { ...useServiceStore.getState().services, events: { record: async () => { throw new Error('offline') } } },
    })
    useSearchStore.setState({ response: response([extractedPermitted]) })
    render()
    fireEvent.click(screen.getByRole('button', { name: 'Pin reference' }))
    await waitFor(() => expect(useSearchStore.getState().pinned).toHaveLength(1))
  })
})

describe('pin notices', () => {
  it('reports revoked pins and can dismiss them', async () => {
    useSearchStore.setState({
      response: response([extractedPermitted]),
      revokedPins: [
        { assetId: 'ls_a', reasons: ['display_not_permitted'] },
        { assetId: 'ls_b', reasons: ['asset_disabled'] },
      ],
    })
    render()
    expect(screen.getByTestId('revoked-pins')).toHaveTextContent('2 pinned references are no longer available')
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
    await waitFor(() => expect(screen.queryByTestId('revoked-pins')).toBeNull())
  })

  it('shows pinned references in their own section', () => {
    useSearchStore.setState({ response: response([extractedPermitted]), pinned: [asset('pinned-1')] })
    render()
    expect(screen.getByRole('heading', { name: 'Pinned' })).toBeInTheDocument()
  })
})

afterEach(() => vi.restoreAllMocks())
