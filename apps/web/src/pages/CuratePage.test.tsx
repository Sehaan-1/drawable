import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { TooltipProvider } from '@radix-ui/react-tooltip'
import { useServiceStore } from '../services/serviceRegistry'
import { makeCandidate } from '../test/candidateFactory'
import type { JSX, ReactNode } from 'react'

const originalFetch = globalThis.fetch

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function makeWrapper(): ({ children }: { children: ReactNode }) => JSX.Element {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <TooltipProvider delayDuration={500}>{children}</TooltipProvider>
        </MemoryRouter>
      </QueryClientProvider>
    )
  }
}

beforeEach(() => {
  // Default to a live API with curation enabled. Individual tests override
  // this (fixture mode, curation off).
  useServiceStore.setState({
    mode: 'live',
    health: {
      mode: 'cpu',
      ready: true,
      message: 'API',
      live: true,
      health: {
        ready: true,
        fixture_mode: true,
        cuda_available: false,
        device: 'cpu',
        gpu_name: null,
        vram_total_mb: null,
        torch_version: null,
        api_version: '0.1.0',
        schema_version: 1,
        preprocessing_version: '1.0.0',
        models: [],
        dataset_version: 'synthetic',
        index_version: 'abc',
        gallery_size: 24,
        disabled_branches: [],
        warmup: 'skipped',
        warnings: [],
        curation_enabled: true,
      },
    },
  })
})

afterEach(() => {
  globalThis.fetch = originalFetch
  vi.restoreAllMocks()
  useServiceStore.setState({ mode: 'probing', health: null })
})

describe('CuratePage', () => {
  it('renders the curator once a candidate is loaded', async () => {
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_test', thumbnail_url: '/api/v1/assets/ls_test/thumbnail', line_art_url: '/api/v1/assets/ls_test/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {
              manga_anime: { reviewed: 0, accepted: 0, rejected: 0, remaining: 1 },
              western_ink: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              realistic_academic: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              cartoon: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              gesture_sketch: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
            },
            by_scope: {
              eye: { reviewed: 0, accepted: 0, rejected: 0, remaining: 1 },
              face_head: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              hair: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              hand: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              foot: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              upper_body_clothing: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              full_body: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              multi_character: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
            },
          }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch

    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await waitFor(() => expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_test'))
    // The image is rendered through the real candidate URL.
    const img = screen.getByTestId('candidate-img') as HTMLImageElement
    expect(img.getAttribute('src')).toBe('/api/v1/assets/ls_test/thumbnail')
  })

  it('shows the empty state when the queue is drained', async () => {
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse({ error: { code: 'queue_empty', message: 'empty' } }, 404),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 24,
            accepted: 24,
            rejected: 0,
            remaining: 0,
            target: 2000,
            by_style: {
              manga_anime: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              western_ink: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              realistic_academic: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              cartoon: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              gesture_sketch: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
            },
            by_scope: {
              eye: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              face_head: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              hair: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              hand: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              foot: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              upper_body_clothing: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              full_body: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
              multi_character: { reviewed: 0, accepted: 0, rejected: 0, remaining: 0 },
            },
          }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await waitFor(() => expect(screen.getByText(/queue empty/i)).toBeInTheDocument())
  })

  it('renders a fixture candidate when the API is offline', async () => {
    useServiceStore.setState({
      mode: 'fixture',
      health: { mode: 'fixture', ready: true, message: 'Procedural references · no backend required' },
    })
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    expect(await screen.findByText('Candidate metadata')).toBeInTheDocument()
    expect(screen.queryByText(/Start the local API to curate/i)).toBeNull()
    expect(screen.getByTestId('inspector-asset-id').textContent).toBe('fixture-curate-01')
  })

  it('disables the page when curation is off and the API is live', async () => {
    useServiceStore.setState({
      mode: 'live',
      health: {
        mode: 'cpu',
        ready: true,
        message: 'API',
        live: true,
        health: {
          ready: true,
          fixture_mode: true,
          cuda_available: false,
          device: 'cpu',
          gpu_name: null,
          vram_total_mb: null,
          torch_version: null,
          api_version: '0.1.0',
          schema_version: 1,
          preprocessing_version: '1.0.0',
          models: [],
          dataset_version: 'synthetic',
          index_version: 'abc',
          gallery_size: 24,
          disabled_branches: [],
          warmup: 'skipped',
          warnings: [],
          curation_enabled: false,
        },
      },
    })
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    expect(await screen.findByText(/Curation mode is off/i)).toBeInTheDocument()
  })

  it('submits a label on the K keyboard shortcut', async () => {
    let labelPayload: unknown = null
    globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_shortcut', thumbnail_url: '/api/v1/assets/ls_shortcut/thumbnail', line_art_url: '/api/v1/assets/ls_shortcut/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      if (url.includes('/curation/labels')) {
        labelPayload = init ? JSON.parse(init.body as string) : null
        return Promise.resolve(
          mockJsonResponse({ id: 1, asset_id: 'ls_shortcut', decision: 'keep', review_state: 'accepted' }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    // Set quality via the shortcut, then submit on K.
    fireEvent.keyDown(window, { key: '3' })
    fireEvent.keyDown(window, { key: 'k' })
    await waitFor(() => expect(labelPayload).toBeTruthy())
    expect(labelPayload).toMatchObject({
      asset_id: 'ls_shortcut',
      expected_review_state: 'unreviewed',
      expected_label_version: 0,
      decision: 'keep',
      quality: 3,
      // The session scopes the server-side skip cursor.
      session_id: expect.any(String),
    })
    // Crops are no longer part of the label — they are immutable derivatives.
    expect(labelPayload).not.toHaveProperty('crop')
    // The session id is also sent when fetching the next candidate.
    const calls = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls
    const nextUrl = calls.map((call) => String(call[0])).find((url) => url.includes('/curation/next'))
    expect(nextUrl).toContain('session_id=')
  })

  it('submits a label on the R keyboard shortcut', async () => {
    let labelPayload: unknown = null
    globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_reject', thumbnail_url: '/api/v1/assets/ls_reject/thumbnail', line_art_url: '/api/v1/assets/ls_reject/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      if (url.includes('/curation/labels')) {
        labelPayload = init ? JSON.parse(init.body as string) : null
        return Promise.resolve(
          mockJsonResponse({ id: 1, asset_id: 'ls_reject', decision: 'reject', review_state: 'rejected' }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    fireEvent.keyDown(window, { key: '1' })
    fireEvent.keyDown(window, { key: 'r' })
    await waitFor(() => expect(labelPayload).toBeTruthy())
    expect(labelPayload).toMatchObject({ decision: 'reject' })
  })

  it('does not submit a keep without a quality', async () => {
    let labelPayload: unknown = null
    globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_noquality', thumbnail_url: '/api/v1/assets/ls_noquality/thumbnail', line_art_url: '/api/v1/assets/ls_noquality/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      if (url.includes('/curation/labels')) {
        labelPayload = init ? JSON.parse(init.body as string) : null
        return Promise.resolve(
          mockJsonResponse({ id: 1, asset_id: 'ls_noquality', decision: 'keep', review_state: 'accepted' }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    fireEvent.keyDown(window, { key: 'k' })
    // Allow any in-flight async work to settle.
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(labelPayload).toBeNull()
  })

  it('renders the keyboard shortcut help HUD', async () => {
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_help', thumbnail_url: '/api/v1/assets/ls_help/thumbnail', line_art_url: '/api/v1/assets/ls_help/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    // The help overlay always lists the core shortcuts.
    expect(screen.getByTestId('kbd-hud')).toBeInTheDocument()
    expect(screen.getByText(/Keep current candidate/i)).toBeInTheDocument()
    expect(screen.getByText(/Reject current candidate/i)).toBeInTheDocument()
    expect(screen.getByText(/Set quality/i)).toBeInTheDocument()
    expect(screen.getByText(/Toggle crop edit/i)).toBeInTheDocument()
  })

  it('flashes the last pressed key in the HUD', async () => {
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_pulse', thumbnail_url: '/api/v1/assets/ls_pulse/thumbnail', line_art_url: '/api/v1/assets/ls_pulse/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    // No pulse yet.
    expect(screen.queryByTestId('kbd-hud-last')).toBeNull()
    // Press a quality key and the pulse appears.
    fireEvent.keyDown(window, { key: '2' })
    expect(await screen.findByTestId('kbd-hud-last')).toHaveTextContent('2')
  })

  it('restores the previous candidate from history', async () => {
    let nextCount = 0
    const candidate = (assetId: string) =>
      makeCandidate({
        asset_id: assetId,
        thumbnail_url: `/api/v1/curation/assets/${assetId}/thumbnail`,
        line_art_url: `/api/v1/curation/assets/${assetId}/line-art`,
      })
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/next')) {
        const id = nextCount === 0 ? 'ls_first' : 'ls_second'
        nextCount += 1
        return Promise.resolve(mockJsonResponse(candidate(id)))
      }
      if (url.includes('/curation/candidates/')) {
        // Previous re-fetches the actual previous candidate by id.
        const id = url.split('/curation/candidates/')[1]?.split('?')[0] ?? ''
        return Promise.resolve(mockJsonResponse(candidate(id)))
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await waitFor(() => expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_first'))
    fireEvent.click(screen.getByTestId('next-candidate'))
    await waitFor(() => expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_second'))
    fireEvent.click(screen.getByTestId('prev-candidate'))
    await waitFor(() => expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_first'))
  })

  it('shows a toast when K is pressed without a quality', async () => {
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/next')) {
        return Promise.resolve(
          mockJsonResponse(makeCandidate({ asset_id: 'ls_toast', thumbnail_url: '/api/v1/assets/ls_toast/thumbnail', line_art_url: '/api/v1/assets/ls_toast/line-art' })),
        )
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({
            reviewed: 0,
            accepted: 0,
            rejected: 0,
            remaining: 2000,
            target: 2000,
            by_style: {},
            by_scope: {},
          }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    expect(screen.queryByTestId('kbd-toast-missing-quality')).toBeNull()
    fireEvent.keyDown(window, { key: 'k' })
    expect(await screen.findByTestId('kbd-toast-missing-quality')).toBeInTheDocument()
  })
  it('skips the candidate without labeling it and advances the queue', async () => {
    let nextCount = 0
    let skipPayload: unknown = null
    globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/curation/queue/skip')) {
        skipPayload = init ? JSON.parse(init.body as string) : null
        return Promise.resolve(
          mockJsonResponse({ session_id: 'sess', cursor_asset_id: 'ls_skip', excluded_asset_id: 'ls_skip', remaining: 1999 }),
        )
      }
      if (url.includes('/curation/labels')) {
        return Promise.reject(new Error('labels must not be called on skip'))
      }
      if (url.includes('/curation/next')) {
        const id = nextCount === 0 ? 'ls_skip' : 'ls_after'
        nextCount += 1
        return Promise.resolve(mockJsonResponse(makeCandidate({ asset_id: id })))
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({ reviewed: 0, accepted: 0, rejected: 0, quarantined: 0, remaining: 2000, target: 2000, by_style: {}, by_scope: {} }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await waitFor(() => expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_skip'))
    fireEvent.keyDown(window, { key: 's' })
    await waitFor(() => expect(skipPayload).toBeTruthy())
    expect(skipPayload).toMatchObject({ asset_id: 'ls_skip', session_id: expect.any(String) })
    // The queue advances to the next candidate after a skip.
    await waitFor(() => expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_after'))
  })

  it('shows a conflict banner when the label version is stale', async () => {
    let reloadCount = 0
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/curation/labels')) {
        return Promise.resolve(
          mockJsonResponse(
            {
              error: {
                code: 'label_version_conflict',
                message: 'label was updated by another reviewer',
                details: {
                  current_label_version: 4,
                  current_review_state: 'accepted',
                  latest_decision: 'keep',
                  latest_reviewer: 'other-curator',
                },
              },
            },
            409,
          ),
        )
      }
      if (url.includes('/curation/candidates/')) {
        reloadCount += 1
        return Promise.resolve(
          makeCandidate({ asset_id: 'ls_conflict', review_state: 'accepted', label_version: 4 }),
        )
      }
      if (url.includes('/curation/next')) {
        return Promise.resolve(mockJsonResponse(makeCandidate({ asset_id: 'ls_conflict' })))
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({ reviewed: 0, accepted: 0, rejected: 0, quarantined: 0, remaining: 2000, target: 2000, by_style: {}, by_scope: {} }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    fireEvent.keyDown(window, { key: '3' })
    fireEvent.keyDown(window, { key: 'k' })
    const banner = await screen.findByTestId('conflict-banner')
    expect(banner.textContent).toContain('4')
    expect(banner.textContent).toContain('accepted')
    // Reloading re-fetches the live candidate by id.
    fireEvent.click(screen.getByTestId('conflict-reload'))
    await waitFor(() => expect(reloadCount).toBeGreaterThan(0))
  })

  it('reveals and adjudicates quarantined records behind the toggle', async () => {
    let adjudicationPayload: unknown = null
    let revealed = false
    const heldEntry = {
      asset_id: 'ls_held',
      review_state: 'quarantined',
      label_version: 2,
      revealed: false,
      thumbnail_url: null,
      line_art_url: null,
      reveal_expires_at: null,
      parent_asset_id: null,
      primary_style: 'manga_anime',
      primary_scope: 'eye',
      source_work_id: 'src_1',
      quality_score: 0.5,
      width: 512,
      height: 512,
      blockers: [],
      derivative_processing_state: null,
      sfw_screening: { method: 'opennsfw2', verdict: 'unsure', confidence: 0.62 },
      sfw_human: null,
    }
    const revealedEntry = {
      ...heldEntry,
      revealed: true,
      thumbnail_url: '/api/v1/curation/quarantine/ls_held/thumbnail',
      reveal_expires_at: '2026-01-01T00:05:00Z',
    }
    globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/curation/quarantine/ls_held/reveal')) {
        revealed = true
        return Promise.resolve(mockJsonResponse(revealedEntry))
      }
      if (url.includes('/curation/sfw/ls_held/adjudication')) {
        adjudicationPayload = init ? JSON.parse(init.body as string) : null
        return Promise.resolve(
          mockJsonResponse({ id: 7, asset_id: 'ls_held', safe: true, reviewer: 'me', decided_at: 'x' }),
        )
      }
      if (url.includes('/curation/quarantine')) {
        return Promise.resolve(mockJsonResponse([revealed ? revealedEntry : heldEntry]))
      }
      if (url.includes('/curation/next')) {
        return Promise.resolve(mockJsonResponse(makeCandidate({ asset_id: 'ls_queue' })))
      }
      if (url.includes('/curation/progress')) {
        return Promise.resolve(
          mockJsonResponse({ reviewed: 0, accepted: 0, rejected: 0, quarantined: 1, remaining: 2000, target: 2000, by_style: {}, by_scope: {} }),
        )
      }
      return Promise.reject(new Error('unexpected URL ' + url))
    }) as unknown as typeof fetch
    const CuratePage = (await import('./CuratePage')).default
    render(<CuratePage />, { wrapper: makeWrapper() })
    await screen.findByTestId('inspector-asset-id')
    // Nothing quarantine-related renders until the toggle is flipped.
    expect(screen.queryByTestId('quarantine-panel')).toBeNull()
    fireEvent.click(screen.getByTestId('quarantine-toggle'))
    const panel = await screen.findByTestId('quarantine-panel')
    expect(panel.textContent).toContain('ls_held')
    // Metadata only: no preview until a deliberate reveal.
    expect(screen.queryByTestId('quarantine-preview-ls_held')).toBeNull()
    fireEvent.click(screen.getByTestId('quarantine-reveal-ls_held'))
    await screen.findByTestId('quarantine-preview-ls_held')
    // Safe adjudication carries the expected label version.
    fireEvent.click(screen.getByTestId('quarantine-safe-ls_held'))
    await waitFor(() => expect(adjudicationPayload).toBeTruthy())
    expect(adjudicationPayload).toMatchObject({ safe: true, expected_label_version: 2 })
  })

  it('cuts and processes a crop derivative from the staged crop', async () => {
    let cropPayload: unknown = null
    const widthDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'naturalWidth')
    const heightDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'naturalHeight')
    Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', { configurable: true, get: () => 512 })
    Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', { configurable: true, get: () => 512 })
    try {
      globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
        if (url.includes('/curation/assets/ls_crop/crops')) {
          cropPayload = init ? JSON.parse(init.body as string) : null
          return Promise.resolve(
            mockJsonResponse({
              asset_id: 'ls_child',
              created: true,
              crop: { x: 51, y: 51, width: 410, height: 410 },
              derivatives_current: false,
              height: 410,
              label_version: 0,
              parent_asset_id: 'ls_crop',
              processing_state: 'pending',
              review_state: 'unreviewed',
              width: 410,
            }),
          )
        }
        if (url.includes('/curation/assets/ls_child/process')) {
          return Promise.resolve(
            mockJsonResponse({
              asset_id: 'ls_child',
              parent_asset_id: 'ls_crop',
              label_version: 0,
              processing_state: 'complete',
              derivatives_current: true,
              attempts: 1,
              enabled: true,
              embedding_status: 'missing',
              measurements: {
                background_coverage: 0.4,
                ink_coverage: 0.1,
                phash: 'abc123',
                quality_score: 0.91,
                text_coverage: 0,
                width: 410,
                height: 410,
              },
              artifact: { kind: 'crop', path: 'derivatives/ls_child.png', sha256: 'deadbeef', created_at: 'x' },
            }),
          )
        }
        if (url.includes('/curation/next')) {
          return Promise.resolve(mockJsonResponse(makeCandidate({ asset_id: 'ls_crop' })))
        }
        if (url.includes('/curation/progress')) {
          return Promise.resolve(
            mockJsonResponse({ reviewed: 0, accepted: 0, rejected: 0, quarantined: 0, remaining: 2000, target: 2000, by_style: {}, by_scope: {} }),
          )
        }
        return Promise.reject(new Error('unexpected URL ' + url))
      }) as unknown as typeof fetch
      const CuratePage = (await import('./CuratePage')).default
      render(<CuratePage />, { wrapper: makeWrapper() })
      await screen.findByTestId('inspector-asset-id')
      // Enter crop mode and let the stage seed the default crop rect.
      fireEvent.click(screen.getByTestId('toggle-crop'))
      const img = await screen.findByTestId('candidate-img')
      fireEvent.load(img)
      await waitFor(() =>
        expect((screen.getByTestId('create-derivative') as HTMLButtonElement).disabled).toBe(false),
      )
      fireEvent.click(screen.getByTestId('create-derivative'))
      await waitFor(() => expect(cropPayload).toBeTruthy())
      expect(cropPayload).toMatchObject({
        crop: { x: expect.any(Number), y: expect.any(Number), width: expect.any(Number), height: expect.any(Number) },
        expected_label_version: 0,
      })
      const toast = await screen.findByTestId('derivative-toast')
      expect(toast.textContent).toContain('ls_child')
      expect(toast.textContent).toContain('quality 0.91')
    } finally {
      if (widthDescriptor) Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', widthDescriptor)
      if (heightDescriptor) Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', heightDescriptor)
    }
  })

  it('surfaces parent-artifact problems when a crop cut fails', async () => {
    const widthDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'naturalWidth')
    const heightDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'naturalHeight')
    Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', { configurable: true, get: () => 512 })
    Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', { configurable: true, get: () => 512 })
    try {
      globalThis.fetch = vi.fn().mockImplementation((url: string) => {
        if (url.includes('/curation/assets/ls_stale/crops')) {
          return Promise.resolve(
            mockJsonResponse(
              {
                error: {
                  code: 'parent_artifact_invalid',
                  message: 'parent artifacts failed verification',
                  details: {
                    problems: ['thumbnail checksum mismatch', 'line art missing'],
                  },
                },
              },
              422,
            ),
          )
        }
        if (url.includes('/curation/next')) {
          return Promise.resolve(mockJsonResponse(makeCandidate({ asset_id: 'ls_stale' })))
        }
        if (url.includes('/curation/progress')) {
          return Promise.resolve(
            mockJsonResponse({ reviewed: 0, accepted: 0, rejected: 0, quarantined: 0, remaining: 2000, target: 2000, by_style: {}, by_scope: {} }),
          )
        }
        return Promise.reject(new Error('unexpected URL ' + url))
      }) as unknown as typeof fetch
      const CuratePage = (await import('./CuratePage')).default
      render(<CuratePage />, { wrapper: makeWrapper() })
      await screen.findByTestId('inspector-asset-id')
      fireEvent.click(screen.getByTestId('toggle-crop'))
      const img = await screen.findByTestId('candidate-img')
      fireEvent.load(img)
      await waitFor(() =>
        expect((screen.getByTestId('create-derivative') as HTMLButtonElement).disabled).toBe(false),
      )
      fireEvent.click(screen.getByTestId('create-derivative'))
      const toast = await screen.findByTestId('derivative-toast')
      expect(toast.textContent).toContain('was not cut')
      expect(toast.textContent).toContain('thumbnail checksum mismatch')
      expect(toast.textContent).toContain('line art missing')
      expect(toast.textContent).toContain('parent was not modified')
      // No conflict banner for a validation failure — it is not a 409.
      expect(screen.queryByTestId('conflict-banner')).toBeNull()
    } finally {
      if (widthDescriptor) Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', widthDescriptor)
      if (heightDescriptor) Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', heightDescriptor)
    }
  })
})
