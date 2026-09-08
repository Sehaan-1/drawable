import type { CurationCandidate } from '../components/CuratePage/types'

/**
 * Builds a fully-granted v2 curation candidate for tests. Overrides are
 * applied last so each test only spells out the fields it cares about.
 */
export function makeCandidate(overrides: Partial<CurationCandidate> = {}): CurationCandidate {
  return {
    asset_id: 'ls_synthetic_ac1f55b7390698a7',
    primary_style: 'manga_anime',
    primary_scope: 'eye',
    secondary_scopes: [],
    person_count: 1,
    person_count_approximate: false,
    width: 256,
    height: 256,
    thumbnail_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/thumbnail',
    line_art_url: '/api/v1/assets/ls_synthetic_ac1f55b7390698a7/line-art',
    origin: 'native_line_art',
    crop: null,
    review_state: 'unreviewed',
    blockers: [],
    quality_score: 0.85,
    sfw_screening: { verdict: 'safe', confidence: 0.99, method: 'source_rating' },
    sfw_human: null,
    permissions: {
      license_id: 'synthetic',
      basis: 'first_party',
      permission_url: null,
      attribution: 'Synthetic fixture',
      attribution_required: true,
    },
    allowed_uses: { display: true, training: true, trace: true },
    learning_split: 'train',
    gallery_member: true,
    gold_member: false,
    source_work_id: 'synthetic-work-000',
    parent_asset_id: null,
    artist_id: null,
    leakage_group_id: null,
    ...overrides,
  }
}
