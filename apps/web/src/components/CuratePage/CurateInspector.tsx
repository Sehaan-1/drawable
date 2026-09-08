/**
 * Right-side inspector for the current candidate.
 *
 * Holds the in-flight review form (primary style override, primary scope,
 * secondary scopes, quality, named blockers, the human SFW decision, and a
 * free-form note). Submission is delegated to the parent: the parent owns the
 * keyboard shortcuts and the React Query mutation, this component just exposes
 * a controlled form so ``K`` / ``R`` / ``1..3`` can pre-fill it before commit.
 *
 * Schema-v2 rules mirrored here so the server never has to bounce a label:
 * - a keep requires a quality, a known primary scope, and zero blockers;
 * - secondary scopes are gallery scopes and never repeat the primary;
 * - the human SFW decision is tri-state: unset, approved, or flagged.
 */

import { useEffect, useState } from 'react'
import { Check, X } from 'lucide-react'
import {
  BLOCKER_TITLES,
  CURATION_BLOCKERS,
  GALLERY_SCOPES,
  PERMISSION_BASIS_TITLES,
  PRIMARY_STYLES,
  SCOPE_LABELS,
  SCOPE_TITLES,
  STYLE_TITLES,
  type CurationBlocker,
  type PrimaryStyle,
  type ScopeLabel,
} from '@drawable/contracts'
import { Button, StatusDot } from '../primitives'
import type { CurationCandidate } from './types'

export interface ReviewFormState {
  primaryStyle: PrimaryStyle
  primaryScope: ScopeLabel
  secondaryScopes: ScopeLabel[]
  quality: 1 | 2 | 3 | null
  note: string
  blockers: CurationBlocker[]
  /** null = no decision recorded, true = human approved safe, false = flagged unsafe. */
  sfwSafe: boolean | null
}

export interface CurateInspectorProps {
  candidate: CurationCandidate | null
  pendingForm: ReviewFormState | null
  onFormChange: (next: ReviewFormState) => void
  onKeep: () => void
  onReject: () => void
  onSnapshot: () => void
  busy: boolean
  snapshotPending: boolean
  disabled?: boolean
}

const QUALITY_DESCRIPTORS: Record<1 | 2 | 3, string> = {
  1: 'Reference quality only',
  2: 'Usable in the gallery',
  3: 'Anchor-quality example',
}

export function defaultForm(candidate: CurationCandidate): ReviewFormState {
  return {
    primaryStyle: candidate.primary_style,
    primaryScope: candidate.primary_scope,
    secondaryScopes: [...(candidate.secondary_scopes ?? [])],
    quality: null,
    note: '',
    blockers: [],
    sfwSafe: null,
  }
}

export function CurateInspector({
  candidate,
  pendingForm,
  onFormChange,
  onKeep,
  onReject,
  onSnapshot,
  busy,
  snapshotPending,
  disabled,
}: CurateInspectorProps) {
  // When the candidate changes, reset the local draft so the inspector
  // doesn't carry reviewer notes from the previous asset forward.
  const [draft, setDraft] = useState<ReviewFormState | null>(pendingForm)
  useEffect(() => {
    setDraft(pendingForm)
  }, [pendingForm, candidate?.asset_id])

  // When the candidate first arrives we also seed a fresh form in the
  // parent so keyboard shortcuts have a target to mutate.
  useEffect(() => {
    if (candidate && !pendingForm) {
      onFormChange(defaultForm(candidate))
    }
  }, [candidate, pendingForm, onFormChange])

  if (!candidate) {
    return (
      <aside className="candidate-inspector">
        <div className="inspector-title">
          <span className="dock-eyebrow">Candidate metadata</span>
          <h2>No candidate</h2>
        </div>
        <p className="scaffold-note">
          The curation queue is empty. Press <kbd>→</kbd> to fetch the next
          asset or export a snapshot to release the work so far.
        </p>
        <div className="review-actions">
          <Button disabled={disabled || snapshotPending} onClick={onSnapshot}>
            Export snapshot
          </Button>
        </div>
      </aside>
    )
  }

  const update = (patch: Partial<ReviewFormState>) => {
    if (!draft) return
    const next = { ...draft, ...patch }
    setDraft(next)
    onFormChange(next)
  }

  const setPrimaryScope = (scope: ScopeLabel) => {
    if (!draft) return
    // The primary scope is exactly one label; secondaries must never repeat
    // it, so switching the primary drops it from the secondary set.
    const secondaryScopes = draft.secondaryScopes.filter((value) => value !== scope)
    update({ primaryScope: scope, secondaryScopes })
  }

  const toggleSecondaryScope = (scope: ScopeLabel) => {
    if (!draft) return
    const has = draft.secondaryScopes.includes(scope)
    const secondaryScopes = has
      ? draft.secondaryScopes.filter((value) => value !== scope)
      : [...draft.secondaryScopes, scope]
    update({ secondaryScopes })
  }

  const toggleBlocker = (blocker: CurationBlocker) => {
    if (!draft) return
    const has = draft.blockers.includes(blocker)
    const blockers = has ? draft.blockers.filter((value) => value !== blocker) : [...draft.blockers, blocker]
    update({ blockers })
  }

  const form = draft ?? defaultForm(candidate)
  const keepBlockedReason = form.quality === null
    ? 'Pick a quality to keep'
    : form.primaryScope === 'unknown'
      ? 'Set a known primary scope to keep'
      : form.blockers.length > 0
        ? 'Blocked assets are rejected, not kept'
        : null
  const canKeep = keepBlockedReason === null && !busy
  const canReject = !busy

  const sfwScreening = candidate.sfw_screening
  const sfwHuman = candidate.sfw_human

  return (
    <aside className="candidate-inspector">
      <div className="inspector-title">
        <span className="dock-eyebrow">Candidate metadata</span>
        <h2 data-testid="inspector-asset-id">{candidate.asset_id}</h2>
      </div>

      <label>
        <span>Primary style</span>
        <select
          value={form.primaryStyle}
          onChange={(event) => update({ primaryStyle: event.target.value as PrimaryStyle })}
          disabled={busy}
          data-testid="style-select"
        >
          {PRIMARY_STYLES.map((style) => (
            <option key={style} value={style}>
              {STYLE_TITLES[style]}
            </option>
          ))}
        </select>
      </label>

      <label>
        <span>Primary scope</span>
        <select
          value={form.primaryScope}
          onChange={(event) => setPrimaryScope(event.target.value as ScopeLabel)}
          disabled={busy}
          data-testid="primary-scope-select"
        >
          {SCOPE_LABELS.map((scope) => (
            <option key={scope} value={scope}>
              {SCOPE_TITLES[scope]}
            </option>
          ))}
        </select>
        <small className="field-hint">Exactly one; “Unknown” keeps the asset provisional</small>
      </label>

      <fieldset>
        <legend>Secondary scopes</legend>
        <div className="scope-chips">
          {GALLERY_SCOPES.filter((scope) => scope !== form.primaryScope).map((scope) => {
            const active = form.secondaryScopes.includes(scope)
            return (
              <button
                key={scope}
                type="button"
                className={`chip ${active ? 'is-active' : ''}`}
                aria-pressed={active}
                onClick={() => toggleSecondaryScope(scope)}
                disabled={busy}
                data-testid={`scope-chip-${scope}`}
              >
                {SCOPE_TITLES[scope]}
              </button>
            )
          })}
        </div>
      </fieldset>

      <fieldset>
        <legend>Quality</legend>
        <div className="segmented-control" role="radiogroup" aria-label="Quality">
          {([1, 2, 3] as const).map((value) => (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={form.quality === value}
              className={form.quality === value ? 'is-active' : ''}
              onClick={() => update({ quality: value })}
              title={QUALITY_DESCRIPTORS[value]}
              disabled={busy}
              data-testid={`quality-${value}`}
            >
              {value}
            </button>
          ))}
        </div>
        <small className="field-hint">
          {form.quality ? QUALITY_DESCRIPTORS[form.quality] : 'Required to keep'}
        </small>
      </fieldset>

      <fieldset className="flag-fieldset">
        <legend>Blockers</legend>
        {CURATION_BLOCKERS.map((blocker) => (
          <label key={blocker} className="checkbox">
            <input
              type="checkbox"
              checked={form.blockers.includes(blocker)}
              onChange={() => toggleBlocker(blocker)}
              disabled={busy}
              data-testid={`blocker-${blocker}`}
            />
            {BLOCKER_TITLES[blocker]}
          </label>
        ))}
        <small className="field-hint">Blocking is terminal: keep is disabled while any blocker is set</small>
      </fieldset>

      <fieldset>
        <legend>Human SFW decision</legend>
        <div className="segmented-control" role="radiogroup" aria-label="Human SFW decision">
          {([
            ['unset', 'Not reviewed'],
            ['safe', 'Safe'],
            ['unsafe', 'Unsafe'],
          ] as const).map(([value, label]) => {
            const active =
              (value === 'unset' && form.sfwSafe === null) ||
              (value === 'safe' && form.sfwSafe === true) ||
              (value === 'unsafe' && form.sfwSafe === false)
            return (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={active}
                className={active ? 'is-active' : ''}
                onClick={() => update({ sfwSafe: value === 'unset' ? null : value === 'safe' })}
                disabled={busy}
                data-testid={`sfw-${value}`}
              >
                {label}
              </button>
            )
          })}
        </div>
        <small className="field-hint">
          Serving requires explicit human approval — an automated “safe” verdict is not enough
        </small>
      </fieldset>

      <label className="note-field">
        <span>Review note</span>
        <textarea
          value={form.note}
          onChange={(event) => update({ note: event.target.value })}
          maxLength={2000}
          placeholder="Optional note…"
          disabled={busy}
          data-testid="note-input"
        />
      </label>

      <dl className="metadata-list">
        <div>
          <dt>Source work</dt>
          <dd data-testid="source-work">{candidate.source_work_id}</dd>
        </div>
        <div>
          <dt>Resolution</dt>
          <dd>
            {candidate.width} × {candidate.height}
          </dd>
        </div>
        <div>
          <dt>Line art</dt>
          <dd>{candidate.origin === 'native_line_art' ? 'Native' : 'Extracted'}</dd>
        </div>
        <div>
          <dt>Quality score</dt>
          <dd>{candidate.quality_score.toFixed(2)}</dd>
        </div>
        <div>
          <dt>SFW check</dt>
          <dd data-testid="sfw-check">
            {sfwHuman ? (
              <>
                <StatusDot tone={sfwHuman.safe ? 'success' : 'error'} />
                {sfwHuman.safe ? 'Human approved' : 'Human flagged'}
              </>
            ) : sfwScreening ? (
              <>
                <StatusDot tone={sfwScreening.verdict === 'safe' ? 'warning' : 'error'} />
                {sfwScreening.verdict === 'safe'
                  ? `Screened safe · ${((sfwScreening.confidence ?? 0) * 100).toFixed(0)}% · needs human review`
                  : sfwScreening.verdict === 'unsure'
                    ? 'Screened unsure · quarantined'
                    : 'Screened unsafe · quarantined'}
              </>
            ) : (
              <>
                <StatusDot tone="warning" />
                No automated screening · needs human review
              </>
            )}
          </dd>
        </div>
        <div>
          <dt>Permission</dt>
          <dd data-testid="permission-basis">
            <StatusDot tone={candidate.permissions.basis === 'unknown' ? 'error' : 'success'} />
            {PERMISSION_BASIS_TITLES[candidate.permissions.basis]}
            {candidate.allowed_uses.display ? ' · display' : ''}
            {candidate.allowed_uses.training ? ' · training' : ''}
            {candidate.allowed_uses.trace ? ' · trace' : ''}
            {!candidate.allowed_uses.display && !candidate.allowed_uses.training && !candidate.allowed_uses.trace
              ? ' · no uses allowed'
              : ''}
          </dd>
        </div>
        <div>
          <dt>Learning split</dt>
          <dd data-testid="learning-split">{candidate.learning_split}</dd>
        </div>
        <div>
          <dt>People</dt>
          <dd>
            {candidate.person_count === null
              ? 'Unknown'
              : `${candidate.person_count}${candidate.person_count_approximate ? ' (approx.)' : ''}`}
          </dd>
        </div>
      </dl>

      <div className="review-actions">
        <Button
          className="button--danger"
          onClick={onReject}
          disabled={!canReject}
          data-testid="reject-button"
        >
          <X size={16} /> Reject <kbd>R</kbd>
        </Button>
        <Button
          className="button--primary"
          onClick={onKeep}
          disabled={!canKeep}
          title={keepBlockedReason ?? undefined}
          data-testid="keep-button"
        >
          <Check size={16} /> Keep <kbd>K</kbd>
        </Button>
      </div>

      <div className="review-actions review-actions--secondary">
        <Button
          onClick={onSnapshot}
          disabled={disabled || snapshotPending}
          data-testid="snapshot-button"
        >
          {snapshotPending ? 'Exporting…' : 'Export snapshot'}
        </Button>
      </div>
    </aside>
  )
}
