/**
 * SFW adjudication panel: the held-records backlog, metadata first.
 *
 * The list deliberately ships **without images**. A held record (quarantined
 * by review state, screened unsafe/unsure, or human-flagged) can only be
 * previewed behind a deliberate, expiring reveal grant — the reveal button
 * is right here, the images appear only after the grant exists, and the
 * grant never applies to the public asset routes.
 *
 * Adjudication is the whole point of the panel: ``Safe`` returns a
 * quarantined record to the review queue on its merits; ``Unsafe``
 * quarantines it and closes every serving surface. Both are explicit human
 * decisions, guarded by the record's current label version.
 */

import { Eye, EyeOff, ShieldCheck, ShieldX } from 'lucide-react'
import {
  STYLE_TITLES,
  SCOPE_TITLES,
  type QuarantineCandidate,
  type SfwVerdict,
} from '@drawable/contracts'
import { Button, StatusDot } from '../primitives'

export interface QuarantinePanelProps {
  entries: QuarantineCandidate[] | null
  loading: boolean
  error: string | null
  onRetry: () => void
  onReveal: (assetId: string) => void
  onAdjudicate: (assetId: string, safe: boolean, expectedLabelVersion: number) => void
  /** Asset with an in-flight reveal or adjudication (disables its buttons). */
  pendingAsset: string | null
}

const VERDICT_TONE: Record<SfwVerdict, 'success' | 'warning' | 'error'> = {
  safe: 'success',
  unsure: 'warning',
  unsafe: 'error',
}

function formatVerdict(entry: QuarantineCandidate): string {
  if (entry.sfw_screening) {
    const confidence =
      entry.sfw_screening.confidence === null || entry.sfw_screening.confidence === undefined
        ? ''
        : ` ${(entry.sfw_screening.confidence * 100).toFixed(0)}%`
    return `${entry.sfw_screening.verdict}${confidence}`
  }
  return 'not screened'
}

export function QuarantinePanel({
  entries,
  loading,
  error,
  onRetry,
  onReveal,
  onAdjudicate,
  pendingAsset,
}: QuarantinePanelProps) {
  if (error) {
    return (
      <section className="candidate-stage candidate-stage--empty" aria-label="Quarantine backlog">
        <div className="candidate-empty-card">
          <EyeOff size={28} />
          <h2>Could not load the quarantine backlog</h2>
          <p>{error}</p>
          <Button onClick={onRetry}>Retry</Button>
        </div>
      </section>
    )
  }

  if (loading && !entries) {
    return (
      <section className="candidate-stage candidate-stage--empty" aria-label="Quarantine backlog">
        <div className="candidate-empty-card">
          <Eye size={28} />
          <p>Loading held records…</p>
        </div>
      </section>
    )
  }

  if (!entries || entries.length === 0) {
    return (
      <section className="candidate-stage candidate-stage--empty" aria-label="Quarantine backlog">
        <div className="candidate-empty-card">
          <ShieldCheck size={28} />
          <h2>Nothing is being held</h2>
          <p>
            No records are quarantined or waiting on an SFW decision. When the
            screen or a curator flags a record, it lands here — metadata only,
            behind a deliberate reveal.
          </p>
        </div>
      </section>
    )
  }

  return (
    <section className="quarantine-panel" aria-label="SFW adjudication backlog" data-testid="quarantine-panel">
      <header className="quarantine-head">
        <ShieldX size={16} />
        <div>
          <h2>SFW adjudication</h2>
          <p>
            {entries.length} held record{entries.length === 1 ? '' : 's'} — metadata only until a
            deliberate reveal. Safe returns a record to review on its merits; Unsafe quarantines it
            everywhere.
          </p>
        </div>
      </header>
      <ul className="quarantine-list">
        {entries.map((entry) => {
          const busy = pendingAsset === entry.asset_id
          return (
            <li key={entry.asset_id} className="quarantine-card" data-testid="quarantine-card">
              <div className="quarantine-card__id">
                <code>{entry.asset_id}</code>
                <span data-testid={`quarantine-state-${entry.asset_id}`}>{entry.review_state}</span>
              </div>
              <dl className="quarantine-card__meta">
                <div>
                  <dt>Screen</dt>
                  <dd data-testid={`quarantine-verdict-${entry.asset_id}`}>
                    <StatusDot tone={entry.sfw_screening ? VERDICT_TONE[entry.sfw_screening.verdict] : 'neutral'} />
                    {formatVerdict(entry)}
                  </dd>
                </div>
                <div>
                  <dt>Human</dt>
                  <dd data-testid={`quarantine-human-${entry.asset_id}`}>
                    {entry.sfw_human
                      ? entry.sfw_human.safe
                        ? `safe · ${entry.sfw_human.reviewer}`
                        : `unsafe · ${entry.sfw_human.reviewer}`
                      : 'no decision'}
                  </dd>
                </div>
                <div>
                  <dt>Style / scope</dt>
                  <dd>
                    {STYLE_TITLES[entry.primary_style]} · {SCOPE_TITLES[entry.primary_scope]}
                  </dd>
                </div>
                <div>
                  <dt>Source work</dt>
                  <dd>{entry.source_work_id}</dd>
                </div>
                {entry.parent_asset_id ? (
                  <div>
                    <dt>Crop of</dt>
                    <dd>
                      <code>{entry.parent_asset_id}</code>
                    </dd>
                  </div>
                ) : null}
              </dl>
              {entry.revealed && entry.thumbnail_url ? (
                <figure className="quarantine-card__preview" data-testid={`quarantine-preview-${entry.asset_id}`}>
                  <img src={entry.thumbnail_url} alt={`Held record ${entry.asset_id} (revealed)`} />
                  <figcaption>Revealed until {entry.reveal_expires_at}</figcaption>
                </figure>
              ) : null}
              <div className="quarantine-card__actions">
                <Button
                  onClick={() => onReveal(entry.asset_id)}
                  disabled={busy}
                  data-testid={`quarantine-reveal-${entry.asset_id}`}
                >
                  <Eye size={14} /> {entry.revealed ? 'Extend reveal' : 'Reveal'}
                </Button>
                <Button
                  className="is-primary"
                  onClick={() => onAdjudicate(entry.asset_id, true, entry.label_version)}
                  disabled={busy}
                  data-testid={`quarantine-safe-${entry.asset_id}`}
                >
                  <ShieldCheck size={14} /> Safe
                </Button>
                <Button
                  className="is-danger"
                  onClick={() => onAdjudicate(entry.asset_id, false, entry.label_version)}
                  disabled={busy}
                  data-testid={`quarantine-unsafe-${entry.asset_id}`}
                >
                  <ShieldX size={14} /> Unsafe
                </Button>
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
