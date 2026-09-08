import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { CurateInspector, type ReviewFormState } from './CurateInspector'
import type { CurationCandidate } from './types'
import { makeCandidate } from '../../test/candidateFactory'

const baseForm: ReviewFormState = {
  primaryStyle: 'manga_anime',
  primaryScope: 'eye',
  secondaryScopes: [],
  quality: null,
  note: '',
  blockers: [],
  sfwSafe: null,
}

describe('CurateInspector', () => {
  it('shows the candidate asset id as the heading', () => {
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={baseForm}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    expect(screen.getByTestId('inspector-asset-id').textContent).toBe('ls_synthetic_ac1f55b7390698a7')
  })

  it('disables the Keep button until a quality is selected', () => {
    const onKeep = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={{ ...baseForm, quality: null }}
        onFormChange={vi.fn()}
        onKeep={onKeep}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('keep-button'))
    expect(onKeep).not.toHaveBeenCalled()
  })

  it('enables Keep after a quality is set and invokes it on click', () => {
    const onKeep = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={{ ...baseForm, quality: 2 }}
        onFormChange={vi.fn()}
        onKeep={onKeep}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('keep-button'))
    expect(onKeep).toHaveBeenCalled()
  })

  it('invokes Reject regardless of quality', () => {
    const onReject = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={baseForm}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={onReject}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('reject-button'))
    expect(onReject).toHaveBeenCalled()
  })

  it('toggles a secondary scope chip and notifies the parent', () => {
    const onFormChange = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={baseForm}
        onFormChange={onFormChange}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('scope-chip-full_body'))
    const last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last).toBeTruthy()
    expect(last?.secondaryScopes).toEqual(['full_body'])
  })

  it('changes the primary scope via the select and drops it from secondaries', () => {
    const onFormChange = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={{ ...baseForm, secondaryScopes: ['face_head'] }}
        onFormChange={onFormChange}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.change(screen.getByTestId('primary-scope-select'), { target: { value: 'face_head' } })
    const last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.primaryScope).toBe('face_head')
    expect(last?.secondaryScopes).toEqual([])
  })

  it('changes the primary style via the select', () => {
    const onFormChange = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={baseForm}
        onFormChange={onFormChange}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.change(screen.getByTestId('style-select'), { target: { value: 'cartoon' } })
    const last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.primaryStyle).toBe('cartoon')
  })

  it('reflects quality selection in the form state', () => {
    const onFormChange = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={baseForm}
        onFormChange={onFormChange}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('quality-3'))
    const last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.quality).toBe(3)
  })

  it('toggles the anatomy blocker and disables Keep while blocked', () => {
    const onFormChange = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={baseForm}
        onFormChange={onFormChange}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('blocker-anatomy'))
    const last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.blockers).toEqual(['anatomy'])
    // A blocked asset can only be rejected, never kept.
    expect(screen.getByTestId('keep-button')).toBeDisabled()
    expect(screen.getByTestId('reject-button')).toBeEnabled()
  })

  it('disables the action buttons while a mutation is busy', () => {
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={{ ...baseForm, quality: 3 }}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy
        snapshotPending={false}
      />,
    )
    expect(screen.getByTestId('keep-button')).toBeDisabled()
    expect(screen.getByTestId('reject-button')).toBeDisabled()
  })

  it('shows an empty-state body when no candidate is loaded', () => {
    render(
      <CurateInspector
        candidate={null}
        pendingForm={null}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    expect(screen.getByText(/curation queue is empty/i)).toBeInTheDocument()
  })
})

describe('CurateInspector v2 keep preconditions', () => {
  it('disables Keep while the primary scope is unknown', () => {
    render(
      <CurateInspector
        candidate={makeCandidate({ primary_scope: 'unknown' })}
        pendingForm={{ ...baseForm, primaryScope: 'unknown', quality: 3 }}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    expect(screen.getByTestId('keep-button')).toBeDisabled()
    fireEvent.change(screen.getByTestId('primary-scope-select'), { target: { value: 'eye' } })
  })

  it('records the human SFW decision as tri-state', () => {
    const onFormChange = vi.fn()
    render(
      <CurateInspector
        candidate={makeCandidate()}
        pendingForm={{ ...baseForm, quality: 3 }}
        onFormChange={onFormChange}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    fireEvent.click(screen.getByTestId('sfw-safe'))
    let last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.sfwSafe).toBe(true)
    fireEvent.click(screen.getByTestId('sfw-unsafe'))
    last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.sfwSafe).toBe(false)
    fireEvent.click(screen.getByTestId('sfw-unset'))
    last = onFormChange.mock.calls.at(-1)?.[0] as ReviewFormState | undefined
    expect(last?.sfwSafe).toBeNull()
  })

  it('renders a human-flagged candidate with an error tone', () => {
    render(
      <CurateInspector
        candidate={makeCandidate({ sfw_human: { safe: false, reviewer: 'local' } })}
        pendingForm={baseForm}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    expect(screen.getByTestId('sfw-check').textContent).toContain('Human flagged')
  })

  it('names the permission basis and allowed uses', () => {
    render(
      <CurateInspector
        candidate={makeCandidate({
          permissions: {
            license_id: 'x',
            basis: 'unknown',
            permission_url: null,
            attribution: null,
            attribution_required: false,
          },
          allowed_uses: { display: false, training: false, trace: false },
        })}
        pendingForm={baseForm}
        onFormChange={vi.fn()}
        onKeep={vi.fn()}
        onReject={vi.fn()}
        onSnapshot={vi.fn()}
        busy={false}
        snapshotPending={false}
      />,
    )
    const text = screen.getByTestId('permission-basis').textContent ?? ''
    expect(text).toContain('Unknown')
    expect(text).toContain('no uses allowed')
  })
})
