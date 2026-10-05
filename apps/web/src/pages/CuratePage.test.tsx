import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import CuratePage from './CuratePage'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  sessionStorage.removeItem('drawable-curation-reviewer')
  window.history.replaceState({}, '', '/curate')
})

function mockPrototypeReview() {
  window.history.replaceState({}, '', '/curate?prototype=1')
  let savedReview: Record<string, unknown> | null = null
  const candidate = {
    asset_id: 'prototype-review-image', source_id: 'anita-dataset', revision: 0, state: 'quarantined',
    width: 256, height: 256, preview_available: true, preview_blocker: null,
    metadata: { authorization_basis: 'owner-local-educational-prototype', learning_split: 'pilot_development', labels: {} },
  }
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (init?.method === 'POST' && path.endsWith('/review')) {
      savedReview = JSON.parse(String(init.body)) as Record<string, unknown>
    }
    const value = path.includes('/curation/candidates')
      ? { schema_version: 1, items: savedReview ? [] : [candidate], next_cursor: null }
      : path.includes('/curation/progress')
        ? { total: 1, pending: savedReview ? 0 : 1, candidate: 0, rejected: 0, reviewed: savedReview ? 1 : 0 }
        : { rubric_id: 'drawable-viewpoint-v1', rubric_sha256: 'a'.repeat(64), rubric: {} }
    return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } }))
  }))
  return () => savedReview
}

describe('local curation workspace', () => {
  it('explains the required rationale and permits rejection without completing keep labels', async () => {
    const getReview = mockPrototypeReview()
    render(<MemoryRouter><CuratePage /></MemoryRouter>)
    const reject = await screen.findByRole('button', { name: /^Reject$/ }) as HTMLButtonElement
    const keep = screen.getByRole('button', { name: /^Keep$/ }) as HTMLButtonElement
    expect(reject.disabled).toBe(true)
    expect(keep.disabled).toBe(true)
    expect(screen.getByText(/Enter a short keep \/ reject rationale above/)).toBeTruthy()

    fireEvent.change(screen.getByRole('textbox', { name: 'Review note' }), { target: { value: 'Additional context' } })
    expect(reject.disabled).toBe(true)
    fireEvent.change(screen.getByRole('textbox', { name: 'Keep / reject rationale (required)' }), { target: { value: 'Image is blurry.' } })
    expect(reject.disabled).toBe(false)
    expect(keep.disabled).toBe(true)
    expect(screen.getByText(/Reject does not require these labels/)).toBeTruthy()
    fireEvent.click(reject)
    await waitFor(() => expect(getReview()).toMatchObject({ decision: 'reject', labels: null, note: 'Image is blurry.\nAdditional context' }))
  })

  it('lists missing keep labels and enables saving only after they are completed', async () => {
    const getReview = mockPrototypeReview()
    render(<MemoryRouter><CuratePage /></MemoryRouter>)
    const keep = await screen.findByRole('button', { name: /^Keep$/ }) as HTMLButtonElement
    fireEvent.change(screen.getByRole('textbox', { name: 'Keep / reject rationale (required)' }), { target: { value: 'Clear head sketch.' } })
    fireEvent.change(screen.getByRole('combobox', { name: 'Style' }), { target: { value: 'cartoon' } })
    fireEvent.change(screen.getByRole('combobox', { name: 'Primary scope' }), { target: { value: 'face_head' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: /Person count/ }), { target: { value: '1' } })
    fireEvent.change(screen.getByRole('combobox', { name: 'Safety' }), { target: { value: 'safe' } })
    fireEvent.click(screen.getByRole('button', { name: '3' }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Useful for character design?' }), { target: { value: 'yes' } })
    expect(keep.disabled).toBe(true)
    expect(screen.getByText(/To keep this image, complete: text coverage \(enter 0 if there is no text\)/)).toBeTruthy()
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Text / balloon coverage (0–1)' }), { target: { value: '0' } })
    expect(keep.disabled).toBe(false)
    expect(screen.queryByText(/To keep this image, complete:/)).toBeNull()

    fireEvent.click(screen.getByRole('checkbox', { name: 'Not applicable' }))
    expect(keep.disabled).toBe(true)
    expect(screen.getByText(/Explain why viewpoint is not applicable \(at least 10 characters\)/)).toBeTruthy()
    fireEvent.change(screen.getByRole('textbox', { name: 'Why is viewpoint not applicable?' }), { target: { value: 'No head angle is visible.' } })
    expect(keep.disabled).toBe(false)
    fireEvent.click(keep)
    await waitFor(() => expect(getReview()).toMatchObject({
      decision: 'keep', labels: { primary_scope: 'face_head', person_count: 1, text_coverage: 0, keep_reject_reason: 'Clear head sketch.' },
      viewpoint: { review_state: 'not_applicable', rationale: 'No head angle is visible.' },
    }))
  })

  it('opens actual prototype previews with a reviewer default and another-batch action', async () => {
    window.history.replaceState({}, '', '/curate?prototype=1')
    const candidate = {
      asset_id: 'prototype-real-image', source_id: 'anita-dataset', revision: 0, state: 'quarantined',
      width: 256, height: 256, preview_available: true, preview_blocker: null,
      metadata: { authorization_basis: 'owner-local-educational-prototype', learning_split: 'pilot_development', labels: {} },
    }
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const path = String(input)
      const value = path.includes('/curation/candidates')
        ? { schema_version: 1, items: [candidate], next_cursor: null }
        : path.includes('/curation/progress')
          ? { schema_version: 1, total: 1, pending: 1, candidate: 0, rejected: 0, reviewed: 0 }
          : { rubric_id: 'drawable-viewpoint-v1', rubric_sha256: 'a'.repeat(64), rubric: {} }
      return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    }))
    render(<MemoryRouter><CuratePage /></MemoryRouter>)
    expect(await screen.findByAltText('Local review asset prototype-real-image')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Load another 24 images' })).toBeTruthy()
    expect((screen.getByRole('textbox', { name: 'Pseudonymous reviewer ID' }) as HTMLInputElement).value).toBe('owner-local')
    expect(screen.queryByText(/non-persistent fixture demonstration/i)).toBeNull()
  })

  it('does not silently fall back to fixture data when the local API is disabled', async () => {
    window.history.replaceState({}, '', '/curate')
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(
      JSON.stringify({ detail: 'Local curation is disabled; set CURATION_MODE=1 explicitly.' }),
      { status: 404, headers: { 'Content-Type': 'application/json' } },
    )))
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter><CuratePage /></MemoryRouter>)
    expect((await screen.findAllByText(/Local curation is disabled/)).length).toBeGreaterThan(0)
    expect(fetchMock).toHaveBeenCalled()
    expect(screen.queryByText(/Fixture preview/)).toBeNull()
    expect(screen.queryByText(/184 \/ 2,000/)).toBeNull()
  })

  it('shows a clearly non-persistent demonstration only with the explicit fixture query', () => {
    window.history.replaceState({}, '', '/curate?mode=fixture')
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    render(<MemoryRouter><CuratePage /></MemoryRouter>)
    expect(screen.getByText(/non-persistent fixture demonstration/i)).toBeTruthy()
    expect(screen.getByText(/Fixture preview · no persistence/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^Keep$/ })).toBeNull()
    expect(fetch).not.toHaveBeenCalled()
  })

  it('keeps unknown person counts blank and allows compositions above the pose cap', async () => {
    window.history.replaceState({}, '', '/curate')
    const candidate = {
      asset_id: 'asset-many-people', source_id: 'test-source', revision: 0, state: 'pending',
      width: null, height: null, preview_available: false, preview_blocker: 'terms_not_reviewed',
      metadata: { group_id: 'work-1', work_id: 'work-1', learning_split: 'unassigned', labels: { person_count: null } },
    }
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const path = String(input)
      const value = path.includes('/curation/candidates')
        ? { schema_version: 1, items: [candidate], next_cursor: null }
        : path.includes('/curation/progress')
          ? { schema_version: 1, total: 1, pending: 1, candidate: 0, rejected: 0, reviewed: 0 }
          : { rubric_id: 'drawable-viewpoint-v1', rubric_sha256: 'a'.repeat(64), rubric: {} }
      return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter><CuratePage /></MemoryRouter>)

    const input = await screen.findByRole('spinbutton', { name: /Person count/ }) as HTMLInputElement
    expect(input.value).toBe('')
    expect(input.max).toBe('')
    const uncertain = screen.getByRole('checkbox', { name: /Person count is uncertain/ }) as HTMLInputElement
    expect(uncertain.checked).toBe(true)
    fireEvent.click(uncertain)
    expect(input.disabled).toBe(false)
    fireEvent.change(input, { target: { value: '12' } })
    expect(input.value).toBe('12')
  })
})
