import {
  PRIMARY_STYLES,
  SCOPE_LABELS,
  type CurationCandidate,
  type CurationProgress,
  type ScopeLabel,
  type StyleBreakdown,
} from '@drawable/contracts'
import { fixtureAssets } from './fixtures'

const emptyBucket = (): StyleBreakdown => ({
  reviewed: 0,
  accepted: 0,
  rejected: 0,
  remaining: 0,
})

const preview = fixtureAssets[1]?.imageUrl ?? ''

/** Offline curator shell: one native fixture so /curate works without the API. */
export const fixtureCurationCandidate: CurationCandidate = {
  asset_id: 'fixture-curate-01',
  primary_style: 'manga_anime',
  scopes: ['eye', 'face_head'],
  width: 240,
  height: 300,
  thumbnail_url: preview,
  line_art_url: preview,
  origin: 'native_line_art',
  crop: null,
  review_state: 'unreviewed',
  quality_score: 0.82,
  sfw_safe: true,
  sfw_confidence: 0.99,
  source_work_id: 'fixture-work-01',
}

export const fixtureCurationProgress: CurationProgress = {
  reviewed: 0,
  accepted: 0,
  rejected: 0,
  remaining: 24,
  target: 2000,
  by_style: Object.fromEntries(
    PRIMARY_STYLES.map((style) => [
      style,
      { ...emptyBucket(), remaining: style === 'manga_anime' ? 8 : 4 },
    ]),
  ),
  by_scope: Object.fromEntries(
    SCOPE_LABELS.filter((scope): scope is Exclude<ScopeLabel, 'unknown'> => scope !== 'unknown').map(
      (scope) => [scope, { ...emptyBucket(), remaining: 2 }],
    ),
  ),
}
