import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { Check, ChevronLeft, ChevronRight, Crop, RotateCcw, X } from 'lucide-react'
import { ResearchShell } from '../components/AppChrome'
import { Button, StatusDot } from '../components/primitives'
import { fixtureAssets } from '../services/fixtures'

type ReviewDecision = 'keep' | 'reject' | 'quarantine'
type Axis = 'horizontal_view' | 'facing' | 'elevation'
type Candidate = {
  asset_id: string
  source_id: string
  revision: number
  state: string
  width: number | null
  height: number | null
  preview_available: boolean
  preview_blocker: string | null
  metadata: Record<string, unknown>
}
type CandidateList = { schema_version: number; items: Candidate[]; next_cursor: string | null }
type Progress = { total: number; pending: number; candidate: number; rejected: number; reviewed: number }
type Rubric = { rubric_id: string; rubric_sha256: string; rubric: Record<string, unknown> }
type SourceYieldQueue = { status: string; sample_sha256: string | null; selected_count: number; sources: { source_id: string; target: number; status: string; blockers: string[] }[]; items: Candidate[]; blockers: string[] }
type CandidateFilter = 'all' | 'pending' | 'candidate' | 'rejected'
type EyeCropYield = '' | 'yes' | 'no' | 'uncertain'
type Selection = { x: number; y: number; width: number; height: number }
type QueueFilters = { sourceId: string; workId: string; candidateType: string }
type SourceBox = { xmin: number; ymin: number; xmax: number; ymax: number }
type SourceAnnotation = { annotation_id: string; source_row_id: string; boxes: { face: SourceBox; body: SourceBox }; source_fields: Record<string, string> }

const SCOPES = ['eye', 'eyebrow', 'mouth', 'face_head', 'hair', 'hand', 'foot', 'upper_body_clothing', 'full_body', 'multi_character']
const STYLES = ['manga_anime', 'western_ink', 'realistic_academic', 'cartoon', 'gesture_sketch']
const AXIS_OPTIONS: Record<Axis, string[]> = {
  horizontal_view: ['front', 'three_quarter', 'profile', 'rear', 'indeterminate'],
  facing: ['image_left', 'image_right', 'center', 'indeterminate'],
  elevation: ['level', 'view_from_above', 'view_from_below', 'indeterminate'],
}

function isFixtureMode() {
  return new URLSearchParams(window.location.search).get('mode') === 'fixture'
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, credentials: 'same-origin', cache: 'no-store' })
  const value = await response.json().catch(() => ({})) as { detail?: string }
  if (!response.ok) throw new Error(value.detail ?? `Local API request failed (${response.status})`)
  return value as T
}

function readReviewerId() {
  try { return sessionStorage.getItem('drawable-curation-reviewer') ?? '' } catch { return '' }
}

function metadataText(candidate: Candidate, key: string, fallback = '—') {
  const value = candidate.metadata[key]
  return typeof value === 'string' || typeof value === 'number' ? String(value) : fallback
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'The local curation request failed.'
}

function cropGeometry(image: HTMLImageElement, selection: Selection) {
  const rect = image.getBoundingClientRect()
  const frame = image.parentElement?.getBoundingClientRect()
  if (!image.naturalWidth || !image.naturalHeight || !rect.width || !rect.height) return null
  const scale = Math.min(rect.width / image.naturalWidth, rect.height / image.naturalHeight)
  const displayWidth = image.naturalWidth * scale
  const displayHeight = image.naturalHeight * scale
  const offsetX = (rect.width - displayWidth) / 2
  const offsetY = (rect.height - displayHeight) / 2
  return {
    left: rect.left - (frame?.left ?? rect.left) + offsetX + selection.x * displayWidth,
    top: rect.top - (frame?.top ?? rect.top) + offsetY + selection.y * displayHeight,
    width: selection.width * displayWidth,
    height: selection.height * displayHeight,
  }
}

export default function CuratePage() {
  const fixtureMode = isFixtureMode()
  const prototypeMode = new URLSearchParams(window.location.search).get('prototype') === '1'
  const fixedYieldMode = new URLSearchParams(window.location.search).get('queue') === 'pilot-v1'
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [progress, setProgress] = useState<Progress | null>(null)
  const [yieldQueue, setYieldQueue] = useState<SourceYieldQueue | null>(null)
  const [rubric, setRubric] = useState<Rubric | null>(null)
  const [filter, setFilter] = useState<CandidateFilter>(fixedYieldMode ? 'all' : 'pending')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(!fixtureMode)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [reviewerId, setReviewerId] = useState(() => readReviewerId() || (prototypeMode ? 'owner-local' : ''))
  const [style, setStyle] = useState('')
  const [scope, setScope] = useState('')
  const [learningSplit, setLearningSplit] = useState('unassigned')
  const [quality, setQuality] = useState(0)
  const [sfw, setSfw] = useState<'' | 'safe' | 'unsafe' | 'uncertain'>('')
  const [characterUseful, setCharacterUseful] = useState<'' | 'yes' | 'no'>('')
  const [personCount, setPersonCount] = useState<number | null>(null)
  const [personCountUnknown, setPersonCountUnknown] = useState(false)
  const [textCoverage, setTextCoverage] = useState<number | null>(null)
  const [queueDraft, setQueueDraft] = useState<QueueFilters>({ sourceId: '', workId: '', candidateType: '' })
  const [queueFilters, setQueueFilters] = useState<QueueFilters>({ sourceId: '', workId: '', candidateType: '' })
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [stagingAssetIds, setStagingAssetIds] = useState<string[]>([])
  const [stagingSnapshotPath, setStagingSnapshotPath] = useState<string | null>(null)
  const [clipped, setClipped] = useState(false)
  const [damaged, setDamaged] = useState(false)
  const [reason, setReason] = useState('')
  const [note, setNote] = useState('')
  const [viewpointReviewed, setViewpointReviewed] = useState(false)
  const [viewpointNotApplicable, setViewpointNotApplicable] = useState(false)
  const [usefulEyeCrop, setUsefulEyeCrop] = useState<EyeCropYield>('')
  const [viewpointReason, setViewpointReason] = useState('')
  const [evidenceBasis, setEvidenceBasis] = useState<'visible_crop' | 'crop_with_parent_context'>('visible_crop')
  const [axes, setAxes] = useState<Record<Axis, string>>({ horizontal_view: 'indeterminate', facing: 'indeterminate', elevation: 'indeterminate' })
  const [cropMode, setCropMode] = useState(false)
  const [cropStart, setCropStart] = useState<{ x: number; y: number } | null>(null)
  const [selection, setSelection] = useState<Selection | null>(null)
  const stageImage = useRef<HTMLImageElement>(null)
  const reviewClock = useRef({ activeMs: 0, startedAt: document.visibilityState === 'visible' ? performance.now() : null as number | null })
  const requestSequence = useRef(0)
  const nextCursorRef = useRef<string | null>(null)

  useEffect(() => {
    const pause = () => {
      const clock = reviewClock.current
      if (clock.startedAt !== null) {
        clock.activeMs += Math.max(0, performance.now() - clock.startedAt)
        clock.startedAt = null
      }
    }
    const resume = () => {
      if (document.visibilityState === 'visible' && reviewClock.current.startedAt === null) {
        reviewClock.current.startedAt = performance.now()
      }
    }
    const onVisibility = () => document.visibilityState === 'visible' ? resume() : pause()
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('blur', pause)
    window.addEventListener('focus', resume)
    return () => {
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('blur', pause)
      window.removeEventListener('focus', resume)
    }
  }, [])

  const refresh = useCallback(async (nextFilter: CandidateFilter = filter, append = false, filters: QueueFilters = queueFilters) => {
    const state = nextFilter
    const query = new URLSearchParams({ limit: '100', state })
    if (filters.sourceId) query.set('source_id', filters.sourceId)
    if (filters.workId.trim()) query.set('work_id', filters.workId.trim())
    if (filters.candidateType) query.set('candidate_type', filters.candidateType)
    if (append && nextCursorRef.current) query.set('cursor', nextCursorRef.current)
    const requestId = ++requestSequence.current
    const [list, summary, versionedRubric] = await Promise.all([
      fixedYieldMode
        ? api<SourceYieldQueue>('/api/v1/curation/source-yield').then((sample) => {
          setYieldQueue(sample)
          return { schema_version: 1, items: sample.items, next_cursor: null } satisfies CandidateList
        })
        : api<CandidateList>(`/api/v1/curation/candidates?${query.toString()}`),
      api<Progress>('/api/v1/curation/progress'),
      api<Rubric>('/api/v1/curation/rubric/viewpoint'),
    ])
    if (requestId !== requestSequence.current) return list.items
    setCandidates((current) => append ? [...current, ...list.items] : list.items)
    nextCursorRef.current = list.next_cursor
    setNextCursor(list.next_cursor)
    setProgress(summary)
    setRubric(versionedRubric)
    if (!append) {
      setSelectedId((current) => current && list.items.some((item) => item.asset_id === current)
        ? current
        : (list.items[0]?.asset_id ?? null))
    }
    return list.items
  }, [filter, fixedYieldMode, nextCursor, queueFilters])

  useEffect(() => {
    if (fixtureMode) return
    let active = true
    setLoading(true)
    refresh(fixedYieldMode ? 'all' : 'pending').then(() => {
      if (active) setError(null)
    }).catch((reasonValue: unknown) => {
      if (active) setError(errorMessage(reasonValue))
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [fixtureMode, fixedYieldMode, refresh])

  const selectedIndex = candidates.findIndex((item) => item.asset_id === selectedId)
  const candidate = selectedIndex >= 0 ? candidates[selectedIndex]! : null
  const fixtureIndex = Math.max(0, fixtureAssets.findIndex((item) => item.id === selectedId))
  const fixtureAsset = fixtureAssets[fixtureIndex] ?? fixtureAssets[0]!
  const visibleCandidates = candidates
  const sourceAnnotations = Array.isArray(candidate?.metadata.character_annotations)
    ? candidate.metadata.character_annotations as SourceAnnotation[]
    : []
  const candidatePermissions = candidate?.metadata.use_permissions as Record<string, boolean> | undefined
  const candidateStageable = Boolean(
    candidate?.preview_available
    && candidate.metadata.review_status === 'approved'
    && candidate.metadata.gallery_eligible_candidate === true
    && candidate.metadata.learning_split === 'pilot_development'
    && candidatePermissions?.local_display === true
    && candidatePermissions?.evaluation === true,
  )

  useEffect(() => {
    if (!candidate) return
    const labels = candidate.metadata.labels as Record<string, unknown> | undefined
    const existingViewpoint = candidate.metadata.viewpoint as Record<string, unknown> | undefined
    const existingYield = candidate.metadata.source_yield_review as Record<string, unknown> | undefined
    setStyle(typeof labels?.style === 'string' ? labels.style : '')
    setScope(typeof labels?.primary_scope === 'string' ? labels.primary_scope : '')
    const split = metadataText(candidate, 'learning_split', 'unassigned')
    setLearningSplit(['unassigned', 'pilot_development', 'train', 'validation'].includes(split) ? split : 'unassigned')
    setQuality(typeof labels?.quality === 'number' ? labels.quality : 0)
    setSfw(labels?.sfw === 'safe' || labels?.sfw === 'unsafe' || labels?.sfw === 'uncertain' ? labels.sfw : '')
    setCharacterUseful(typeof labels?.character_useful === 'boolean' ? (labels.character_useful ? 'yes' : 'no') : '')
    setPersonCount(typeof labels?.person_count === 'number' ? labels.person_count : null)
    setPersonCountUnknown(labels?.person_count === null)
    setTextCoverage(typeof labels?.text_coverage === 'number' ? labels.text_coverage : null)
    setClipped(labels?.clipped === true)
    setDamaged(labels?.damaged === true)
    setReason(typeof labels?.keep_reject_reason === 'string' ? labels.keep_reject_reason : '')
    setNote('')
    setViewpointReviewed(existingViewpoint?.review_state === 'reviewed')
    setViewpointNotApplicable(existingViewpoint?.review_state === 'not_applicable')
    setUsefulEyeCrop(existingYield?.useful_eye_crop === 'yes' || existingYield?.useful_eye_crop === 'no' || existingYield?.useful_eye_crop === 'uncertain' ? existingYield.useful_eye_crop : '')
    setViewpointReason(typeof existingViewpoint?.rationale === 'string' ? existingViewpoint.rationale : '')
    setEvidenceBasis(existingViewpoint?.evidence_basis === 'crop_with_parent_context' ? 'crop_with_parent_context' : 'visible_crop')
    setAxes({
      horizontal_view: typeof existingViewpoint?.horizontal_view === 'string' ? existingViewpoint.horizontal_view : 'indeterminate',
      facing: typeof existingViewpoint?.facing === 'string' ? existingViewpoint.facing : 'indeterminate',
      elevation: typeof existingViewpoint?.elevation === 'string' ? existingViewpoint.elevation : 'indeterminate',
    })
    setCropMode(false)
    setSelection(null)
    reviewClock.current = { activeMs: 0, startedAt: document.visibilityState === 'visible' ? performance.now() : null }
  }, [candidate?.asset_id, candidate?.revision])

  function updateFilter(value: CandidateFilter) {
    setFilter(value)
    setLoading(true)
    setError(null)
    refresh(value).catch((reasonValue: unknown) => setError(errorMessage(reasonValue))).finally(() => setLoading(false))
  }

  function applyQueueFilters() {
    setQueueFilters(queueDraft)
    setCandidates([])
    setSelectedId(null)
    nextCursorRef.current = null
    setNextCursor(null)
    setLoading(true)
    setError(null)
    refresh(filter, false, queueDraft).catch((reasonValue: unknown) => setError(errorMessage(reasonValue))).finally(() => setLoading(false))
  }

  const missingKeepLabels = [
    !style && 'style',
    !scope && 'primary scope',
    !(personCountUnknown || (personCount !== null && personCount >= 1)) && 'person count (or mark it uncertain)',
    !sfw && 'safety',
    quality < 1 && 'reference quality',
    !characterUseful && 'character-design usefulness',
    (textCoverage === null || !Number.isFinite(textCoverage)) && 'text coverage (enter 0 if there is no text)',
  ].filter((value): value is string => typeof value === 'string')
  const labelsComplete = missingKeepLabels.length === 0
  const reviewBlocker = busy
    ? 'Saving your changes…'
    : !rubric
      ? 'Wait for the local review form to finish loading.'
      : !reviewerId.trim()
        ? 'Enter your reviewer ID in the left panel to enable the review buttons.'
        : !reason.trim()
          ? 'Enter a short keep / reject rationale above to enable the review buttons. The review note is optional.'
          : viewpointNotApplicable && viewpointReason.trim().length < 10
            ? 'Explain why viewpoint is not applicable (at least 10 characters) to enable the review buttons.'
            : null

  function moveCandidate(offset: number) {
    if (!visibleCandidates.length || selectedIndex < 0) return
    const next = Math.max(0, Math.min(visibleCandidates.length - 1, selectedIndex + offset))
    setSelectedId(visibleCandidates[next]?.asset_id ?? null)
  }

  function localPoint(event: ReactPointerEvent<HTMLDivElement>) {
    const image = stageImage.current
    if (!image?.naturalWidth || !image.naturalHeight) return null
    const rect = image.getBoundingClientRect()
    const scale = Math.min(rect.width / image.naturalWidth, rect.height / image.naturalHeight)
    const displayWidth = image.naturalWidth * scale
    const displayHeight = image.naturalHeight * scale
    const offsetX = (rect.width - displayWidth) / 2
    const offsetY = (rect.height - displayHeight) / 2
    const x = (event.clientX - rect.left - offsetX) / displayWidth
    const y = (event.clientY - rect.top - offsetY) / displayHeight
    if (x < 0 || x > 1 || y < 0 || y > 1) return null
    return { x, y }
  }

  function pointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!cropMode) return
    const point = localPoint(event)
    if (!point) return
    event.currentTarget.setPointerCapture(event.pointerId)
    setCropStart(point)
    setSelection({ ...point, width: 0, height: 0 })
  }

  function pointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!cropMode || !cropStart) return
    const point = localPoint(event)
    if (!point) return
    setSelection({
      x: Math.min(cropStart.x, point.x), y: Math.min(cropStart.y, point.y),
      width: Math.abs(point.x - cropStart.x), height: Math.abs(point.y - cropStart.y),
    })
  }

  function pointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (!cropMode || !cropStart) return
    pointerMove(event)
    setCropStart(null)
  }

  async function saveCrop() {
    if (!candidate || !selection || !reviewerId.trim()) return
    setBusy(true)
    setError(null)
    try {
      const created = await api<{ asset_id: string }>('/api/v1/curation/assets/' + encodeURIComponent(candidate.asset_id) + '/crops', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected_revision: candidate.revision,
          reviewer_id: reviewerId.trim(),
          idempotency_key: crypto.randomUUID(),
          ...selection,
        }),
      })
      await refresh('pending')
      setFilter('pending')
      setSelectedId(created.asset_id)
      setCropMode(false)
      setSelection(null)
    } catch (reasonValue) {
      setError(errorMessage(reasonValue))
    } finally { setBusy(false) }
  }

  async function createM0StagingSnapshot() {
    if (!rubric || !reviewerId.trim() || stagingAssetIds.length < 1 || stagingAssetIds.length > 32) return
    setBusy(true)
    setError(null)
    try {
      const result = await api<{ snapshot_id: string }>('/api/v1/curation/snapshots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected_rubric_sha256: rubric.rubric_sha256,
          requester_id: reviewerId.trim(),
          idempotency_key: crypto.randomUUID(),
          asset_ids: stagingAssetIds,
        }),
      })
      setStagingSnapshotPath(`reports/private/curation-snapshots/${result.snapshot_id}.json`)
    } catch (reasonValue) {
      setError(errorMessage(reasonValue))
    } finally { setBusy(false) }
  }

  async function nextPrototypeBatch() {
    setBusy(true)
    setError(null)
    try {
      await api('/api/v1/curation/prototype/batch?limit=24', { method: 'POST' })
      setFilter('pending')
      await refresh('pending')
    } catch (reasonValue) { setError(errorMessage(reasonValue)) }
    finally { setBusy(false) }
  }

  function toggleStagingAsset(assetId: string) {
    setStagingAssetIds((current) => current.includes(assetId)
      ? current.filter((item) => item !== assetId)
      : current.length >= 32 ? current : [...current, assetId])
    setStagingSnapshotPath(null)
  }

  async function submitReview(nextDecision: ReviewDecision) {
    if (!candidate || !rubric || !reviewerId.trim()) {
      setError('Enter a pseudonymous reviewer ID before saving a review.')
      return
    }
    if (fixedYieldMode && !yieldQueue?.sample_sha256) {
      setError('The fixed source-yield sample is unavailable or stale; refresh it before reviewing.')
      return
    }
    if (fixedYieldMode && usefulEyeCrop === '') {
      setError('Record whether this candidate can produce a useful eye/detail crop before saving the fixed-sample review.')
      return
    }
    if (fixedYieldMode && !viewpointReviewed && !viewpointNotApplicable) {
      setError('Review the visible viewpoint or mark it not applicable before saving the fixed-sample review.')
      return
    }
    const permissions = candidate.metadata.use_permissions as Record<string, boolean> | undefined
    const sourceRights = candidate.metadata.rights_status
    const decisionRights = sourceRights === 'approved' ? 'approved' : sourceRights === 'rejected' ? 'rejected' : 'review_required'
    const previousLabels = candidate.metadata.labels as Record<string, unknown> | undefined
    if (nextDecision === 'keep' && !labelsComplete) {
      setError('Choose every required label before keeping this reference.')
      return
    }
    const viewpoint = viewpointNotApplicable
      ? {
          schema_version: 1, rubric_id: rubric.rubric_id, rubric_sha256: rubric.rubric_sha256,
          review_state: 'not_applicable', evidence_basis: 'not_applicable',
          horizontal_view: 'not_applicable', facing: 'not_applicable', elevation: 'not_applicable',
          reviewer_id: reviewerId.trim(), reviewed_at: new Date().toISOString(),
          rationale: viewpointReason.trim().length >= 10 ? viewpointReason.trim() : 'This asset does not show an applicable head or feature view.',
        }
      : viewpointReviewed
        ? {
            schema_version: 1, rubric_id: rubric.rubric_id, rubric_sha256: rubric.rubric_sha256,
            review_state: 'reviewed', evidence_basis: evidenceBasis,
            ...axes, reviewer_id: reviewerId.trim(), reviewed_at: new Date().toISOString(), rationale: note.trim() || reason.trim(),
          }
        : {
            schema_version: 1, rubric_id: rubric.rubric_id, rubric_sha256: rubric.rubric_sha256,
            review_state: 'unreviewed', evidence_basis: 'visible_crop',
            horizontal_view: 'indeterminate', facing: 'indeterminate', elevation: 'indeterminate',
          }
    setBusy(true)
    setError(null)
    try {
      sessionStorage.setItem('drawable-curation-reviewer', reviewerId.trim())
      await api(`/api/v1/curation/assets/${encodeURIComponent(candidate.asset_id)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected_revision: candidate.revision,
          reviewer_id: reviewerId.trim(),
          idempotency_key: crypto.randomUUID(),
          decision: nextDecision,
          rights_status: decisionRights,
          license_id: typeof candidate.metadata.license_id === 'string' ? candidate.metadata.license_id : null,
          attribution_text: typeof candidate.metadata.attribution_text === 'string' ? candidate.metadata.attribution_text : null,
          terms_snapshot_sha256: typeof candidate.metadata.terms_snapshot_sha256 === 'string' ? candidate.metadata.terms_snapshot_sha256 : null,
          use_permissions: permissions ?? {},
          labels: labelsComplete ? {
            style,
            primary_scope: scope,
            secondary_scopes: Array.isArray(previousLabels?.secondary_scopes) ? previousLabels.secondary_scopes : [],
            person_count: personCountUnknown ? null : personCount,
            sfw,
            quality,
            character_useful: characterUseful === 'yes',
            text_coverage: textCoverage ?? 0,
            clipped,
            damaged,
            duplicate_of: typeof previousLabels?.duplicate_of === 'string' ? previousLabels.duplicate_of : null,
            review_seconds: Math.max(1, Math.round((reviewClock.current.activeMs + (reviewClock.current.startedAt === null ? 0 : performance.now() - reviewClock.current.startedAt)) / 1000)),
            keep_reject_reason: reason.trim() || (nextDecision === 'keep' ? 'Useful reviewed reference.' : 'Rejected during review.'),
          } : null,
          learning_split: learningSplit,
          note: [reason.trim(), note.trim()].filter(Boolean).join('\n'),
          viewpoint,
          source_yield_judgment: fixedYieldMode ? {
            sample_sha256: yieldQueue!.sample_sha256!,
            useful_eye_crop: usefulEyeCrop,
            review_seconds: Math.max(1, Math.round((reviewClock.current.activeMs + (reviewClock.current.startedAt === null ? 0 : performance.now() - reviewClock.current.startedAt)) / 1000)),
          } : null,
        }),
      })
      const refreshed = await refresh(filter)
      const currentIndex = refreshed.findIndex((item) => item.asset_id === candidate.asset_id)
      const next = refreshed[currentIndex + 1] ?? refreshed[currentIndex - 1]
      setSelectedId(next?.asset_id ?? null)
    } catch (reasonValue) {
      setError(errorMessage(reasonValue))
    } finally { setBusy(false) }
  }

  if (fixtureMode) {
    return (
      <ResearchShell eyebrow="Dataset workspace" title="Curation" badge="Fixture preview · no persistence">
        <div className="curation-fixture-warning" role="status">This is a non-persistent fixture demonstration. It does not write judgments, change dataset files, or represent reviewed research data.</div>
        <div className="curation-layout">
          <aside className="research-sidebar">
            <div className="progress-block"><span>Fixture items</span><strong>{fixtureAssets.length} examples</strong><div><i style={{ width: '30%' }} /></div></div>
            <p className="curation-note">Fixture counts and labels are illustrative only.</p>
          </aside>
          <section className="candidate-stage">
            <div className="candidate-toolbar"><span>Example {fixtureIndex + 1} · not persisted</span></div>
            <div className="candidate-image"><img src={fixtureAsset.imageUrl} alt={fixtureAsset.title} /></div>
            <div className="candidate-nav"><Button disabled={fixtureIndex === 0} onClick={() => setSelectedId(fixtureAssets[fixtureIndex - 1]?.id ?? null)}><ChevronLeft size={16} />Previous</Button><span>{fixtureIndex + 1} of {fixtureAssets.length}</span><Button disabled={fixtureIndex === fixtureAssets.length - 1} onClick={() => setSelectedId(fixtureAssets[fixtureIndex + 1]?.id ?? null)}>Next<ChevronRight size={16} /></Button></div>
          </section>
          <aside className="candidate-inspector"><div className="inspector-title"><span className="dock-eyebrow">Fixture example</span><h2>{fixtureAsset.title}</h2></div><p className="curation-note">For a real local review, remove <code>?mode=fixture</code> and start the API with curation explicitly enabled.</p></aside>
        </div>
      </ResearchShell>
    )
  }

  return (
    <ResearchShell eyebrow="Dataset workspace" title={prototypeMode ? 'Review reference images' : 'Curation'} badge={prototypeMode ? 'Local educational prototype' : 'Local API · review only'}>
      <div className="curation-layout">
        <aside className="research-sidebar">
          <div className="progress-block">
            <span>Local review ledger</span>
            <strong>{progress ? `${progress.reviewed} reviewed · ${progress.total} cataloged` : 'Loading local records'}</strong>
            <div><i style={{ width: `${progress?.total ? Math.min(100, 100 * progress.reviewed / progress.total) : 0}%` }} /></div>
          </div>
          {prototypeMode ? <section className="yield-sample-summary" aria-label="Educational image review">
            <strong>Review the downloaded sketches</strong>
            <p>Inspect the image, choose its labels, then keep or reject it. Keep useful head and upper-body references first; create eye/detail crops where suitable.</p>
            <Button className="button--wide" disabled={busy || loading} onClick={() => void nextPrototypeBatch()}>Load another 24 images</Button>
          </section> : null}
          {fixedYieldMode ? <section className="yield-sample-summary" aria-label="Fixed source-yield sample">
            <span className="dock-eyebrow">Fixed pilot-v1 queue</span>
            <strong>{yieldQueue?.items.length ?? 0} of {yieldQueue?.selected_count ?? 230} selected candidates</strong>
            <p>This queue is bound to the recorded pilot-v1 sample. Review each item once and keep measured active time running while you work.</p>
            {yieldQueue?.blockers.map((item) => <small className="curation-error" key={item}>{item}</small>)}
            <ul>{yieldQueue?.sources.map((source) => <li key={source.source_id}><span>{source.source_id}</span><small>{source.status} · target {source.target}{source.blockers.length ? ` · ${source.blockers.join(', ')}` : ''}</small></li>)}</ul>
          </section> : <>
            <label className="curation-filter"><span>Candidate state</span><select value={filter} onChange={(event) => updateFilter(event.target.value as CandidateFilter)}><option value="pending">Needs review</option><option value="candidate">Kept</option><option value="rejected">Rejected</option><option value="all">All unlocked</option></select></label>
            <label className="curation-filter"><span>Source</span><select value={queueDraft.sourceId} onChange={(event) => setQueueDraft((current) => ({ ...current, sourceId: event.target.value }))}><option value="">All sources</option><option value="anita-dataset">AnitaDataset</option><option value="emoreccom-panels">EmoRecCom</option><option value="the-met-open-access">The Met</option><option value="aic-open-access">Art Institute of Chicago</option></select></label>
            <label className="curation-filter"><span>Work ID</span><input value={queueDraft.workId} onChange={(event) => setQueueDraft((current) => ({ ...current, workId: event.target.value }))} placeholder="Optional work or series" /></label>
            <label className="curation-filter"><span>Image type</span><select value={queueDraft.candidateType} onChange={(event) => setQueueDraft((current) => ({ ...current, candidateType: event.target.value }))}><option value="">All types</option><option value="sketch">Native sketch</option><option value="panel_with_official_character_boxes">Comic panel with annotations</option><option value="color">Color frame</option><option value="composition">Composition frame</option><option value="manual_detail_crop">Manual crop</option></select></label>
            <Button className="button--wide" disabled={loading} onClick={applyQueueFilters}>Apply filters</Button>
          </>}
          <nav className="filter-list" aria-label="Current review totals">
            <div><span><StatusDot tone="warning" />Needs review</span><b>{progress?.pending ?? '—'}</b></div>
            <div><span>Kept</span><b>{progress?.candidate ?? '—'}</b></div>
            <div><span>Rejected</span><b>{progress?.rejected ?? '—'}</b></div>
          </nav>
          <section className="m0-staging-controls" aria-label="M0 staging gallery selection">
            <span className="dock-eyebrow">{prototypeMode ? 'Reviewed prototype batch' : 'M0 staging only'}</span>
            <strong>{stagingAssetIds.length} / 32 assets selected</strong>
            <small>{prototypeMode ? 'Select useful kept images to save a batch for the next local model experiment.' : 'Every item must have a reviewed label, approved local display, evaluation permission, and pilot-development family assignment.'}</small>
            <Button className="button--wide button--primary" disabled={busy || !reviewerId.trim() || !stagingAssetIds.length || !rubric} onClick={() => void createM0StagingSnapshot()}>Create review-only snapshot</Button>
            {stagingSnapshotPath ? <code>{stagingSnapshotPath}</code> : null}
          </section>
          <label className="reviewer-field"><span>Pseudonymous reviewer ID</span><input value={reviewerId} onChange={(event) => setReviewerId(event.target.value)} placeholder="e.g. reviewer-01" autoComplete="off" /></label>
          <div className="candidate-queue" aria-label="Candidate list">
            {loading ? <span className="curation-note">Loading…</span> : null}
            {visibleCandidates.map((item) => <button key={item.asset_id} className={item.asset_id === selectedId ? 'is-active' : ''} onClick={() => setSelectedId(item.asset_id)}><span>{item.asset_id.slice(0, 18)}</span><small>{item.source_id} · {fixedYieldMode ? item.metadata.review_status === 'approved' ? 'reviewed' : 'pending' : `r${item.revision}`}</small></button>)}
            {!loading && visibleCandidates.length === 0 ? <span className="curation-note">No candidates in this view.</span> : null}
            {nextCursor ? <Button className="button--wide" disabled={loading} onClick={() => { setLoading(true); refresh(filter, true).catch((reasonValue: unknown) => setError(errorMessage(reasonValue))).finally(() => setLoading(false)) }}>Load next 100</Button> : null}
          </div>
        </aside>

        <section className="candidate-stage" aria-label="Reference review">
          <div className="candidate-toolbar">
            <Button onClick={() => { setCropMode((value) => !value); setSelection(null) }} disabled={!candidate?.preview_available || busy}><Crop size={15} />{cropMode ? 'Cancel crop' : 'Create detail crop'}</Button>
            {selection ? <Button onClick={() => setSelection(null)} disabled={busy}><RotateCcw size={15} />Reset selection</Button> : null}
            <span>{candidate ? `${candidate.asset_id} · revision ${candidate.revision}` : 'No selected asset'}</span>
          </div>
          {candidate?.preview_available ? (
            <div className={`candidate-image ${cropMode ? 'is-cropping' : ''}`} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp}>
              <img ref={stageImage} src={`/api/v1/curation/assets/${encodeURIComponent(candidate.asset_id)}/preview`} alt={`Local review asset ${candidate.asset_id}`} draggable={false} />
              {sourceAnnotations.flatMap((annotation) => (['body', 'face'] as const).map((kind) => {
                const box = annotation.boxes?.[kind]
                if (!box || !candidate.width || !candidate.height || !stageImage.current) return null
                const boxSelection = { x: box.xmin / candidate.width, y: box.ymin / candidate.height, width: (box.xmax - box.xmin) / candidate.width, height: (box.ymax - box.ymin) / candidate.height }
                return <div key={`${annotation.annotation_id}-${kind}`} aria-label={`Source ${kind} box for annotation ${annotation.source_row_id}`} className={`source-box source-box--${kind}`} style={cropGeometry(stageImage.current, boxSelection) ?? undefined}><span>{kind} · row {annotation.source_row_id}</span></div>
              }))}
              {cropMode && selection && stageImage.current ? <div className="crop-selection" aria-label="Selected crop region" style={cropGeometry(stageImage.current, selection) ?? undefined} /> : null}
              {cropMode ? <span className="crop-hint">Drag a rectangle. Crops are saved as new local assets; the source is never changed.</span> : null}
            </div>
          ) : (
            <div className="candidate-image candidate-image--blocked" role="status">
              <div><StatusDot tone="warning" /><h2>{candidate ? 'Preview unavailable' : loading ? 'Loading candidates' : 'No candidate selected'}</h2>
                <p>{candidate?.preview_blocker === 'source_rights_not_approved' || candidate?.preview_blocker === 'source_terms_require_review'
                  ? 'The source and per-work terms do not currently authorize local display. Review and record the official terms before opening image pixels.'
                  : candidate?.preview_blocker === 'local_display_permission_not_granted'
                    ? 'Current source terms do not grant local display for this asset.'
                    : error ?? (candidate ? 'This item is not permitted for image preview.' : 'Choose an unlocked catalog item from the queue.')}</p>
              </div>
            </div>
          )}
          <div className="candidate-nav">
            <Button disabled={selectedIndex <= 0} onClick={() => moveCandidate(-1)}><ChevronLeft size={16} />Previous</Button>
            <span>{candidate ? `${selectedIndex + 1} of ${candidates.length}` : '—'}</span>
            {cropMode && candidate?.preview_available ? <Button className="button--primary" disabled={!selection || busy || !reviewerId.trim()} onClick={saveCrop}><Crop size={15} />Save crop</Button> : <Button disabled={selectedIndex < 0 || selectedIndex >= candidates.length - 1} onClick={() => moveCandidate(1)}>Next<ChevronRight size={16} /></Button>}
          </div>
          <div className="curation-status" aria-live="polite">{error ? <span className="curation-error">{error}</span> : loading ? 'Connecting to the local curation API…' : candidate ? `Source permissions: ${metadataText(candidate, 'rights_status', 'not recorded')} · group ${metadataText(candidate, 'group_id', 'unassigned')}` : 'No local candidate records are available.'}</div>
        </section>

        <aside className="candidate-inspector">
          <div className="inspector-title"><span className="dock-eyebrow">Review &amp; provenance</span><h2>{candidate?.source_id ?? 'No asset selected'}</h2></div>
          {candidate ? <>
            <dl className="metadata-list">
              {candidate.metadata.title ? <div><dt>Artwork</dt><dd>{metadataText(candidate, 'title')}</dd></div> : null}
              {candidate.metadata.artist ? <div><dt>Artist</dt><dd>{metadataText(candidate, 'artist')}</dd></div> : null}
              {candidate.metadata.medium ? <div><dt>Medium</dt><dd>{metadataText(candidate, 'medium')}</dd></div> : null}
              {candidate.metadata.date ? <div><dt>Date</dt><dd>{metadataText(candidate, 'date')}</dd></div> : null}
              <div><dt>Source family</dt><dd>{metadataText(candidate, 'group_id')}</dd></div>
              <div><dt>Work</dt><dd>{metadataText(candidate, 'work_id')}</dd></div>
              {candidate.metadata.provisional_book_prefix ? <div><dt>Book identity</dt><dd>{candidate.metadata.book_identity_verified === true ? `COMICS book ${metadataText(candidate, 'comic_no')}${candidate.metadata.book_title ? ` · ${metadataText(candidate, 'book_title')}` : ''} · crosswalk verified` : `Filename prefix ${String(candidate.metadata.provisional_book_prefix)} · not verified`}</dd></div> : null}
              {!prototypeMode ? <div><dt>Learning split</dt><dd>{metadataText(candidate, 'learning_split', 'unassigned')}</dd></div> : null}
              <div><dt>Dimensions</dt><dd>{candidate.width ?? 'unknown'} × {candidate.height ?? 'unknown'}</dd></div>
              {!prototypeMode ? <div><dt>Rights state</dt><dd>{metadataText(candidate, 'rights_status', 'review_required')}</dd></div> : null}
              <div><dt>License</dt><dd>{metadataText(candidate, 'license_id')}</dd></div>
              {!prototypeMode ? <div><dt>Permission</dt><dd>Display {candidatePermissions?.local_display ? 'yes' : 'no'} · evaluation {candidatePermissions?.evaluation ? 'yes' : 'no'} · derive {candidatePermissions?.derive ? 'yes' : 'no'}</dd></div> : null}
              {candidate.metadata.parent_asset_id ? <div><dt>Crop parent</dt><dd>{String(candidate.metadata.parent_asset_id)}</dd></div> : null}
            </dl>
            <section className="m0-staging-item">
              <Button disabled={busy || (!candidateStageable && !stagingAssetIds.includes(candidate.asset_id)) || (!stagingAssetIds.includes(candidate.asset_id) && stagingAssetIds.length >= 32)} onClick={() => toggleStagingAsset(candidate.asset_id)}>
                {stagingAssetIds.includes(candidate.asset_id) ? prototypeMode ? 'Remove from reviewed batch' : 'Remove from M0 staging' : prototypeMode ? 'Add to reviewed batch' : 'Add to M0 staging'}
              </Button>
              {!candidateStageable && !stagingAssetIds.includes(candidate.asset_id) ? <small>{prototypeMode ? 'Keep this image with suitable, complete labels before adding it to your reviewed batch.' : 'This item cannot enter the staging set until its source permissions, review, and development split pass.'}</small> : null}
            </section>
            {sourceAnnotations.length ? <section className="source-annotations" aria-label="Published character annotations"><h3>Published character boxes</h3><p className="curation-note">These are face/body proposals from the source. They are not accepted Drawable labels.</p>{sourceAnnotations.map((annotation) => <div key={annotation.annotation_id}><span>Source row {annotation.source_row_id}</span><Button disabled={!candidate.preview_available || busy} onClick={() => { const box = annotation.boxes.face; if (!candidate.width || !candidate.height) return; setCropMode(true); setSelection({ x: box.xmin / candidate.width, y: box.ymin / candidate.height, width: (box.xmax - box.xmin) / candidate.width, height: (box.ymax - box.ymin) / candidate.height }) }}>Propose face crop</Button><Button disabled={!candidate.preview_available || busy} onClick={() => { const box = annotation.boxes.body; if (!candidate.width || !candidate.height) return; setCropMode(true); setSelection({ x: box.xmin / candidate.width, y: box.ymin / candidate.height, width: (box.xmax - box.xmin) / candidate.width, height: (box.ymax - box.ymin) / candidate.height }) }}>Propose body crop</Button></div>)}</section> : null}
            <label><span>Style</span><select value={style} onChange={(event) => setStyle(event.target.value)}><option value="">Choose a reviewed style</option>{STYLES.map((value) => <option key={value} value={value}>{value.replaceAll('_', ' ')}</option>)}</select></label>
            <label><span>Primary scope</span><select value={scope} onChange={(event) => setScope(event.target.value)}><option value="">Choose a visible scope</option>{SCOPES.map((value) => <option key={value} value={value}>{value.replaceAll('_', ' ')}</option>)}</select></label>
            {!prototypeMode ? <><label><span>Source-family split (atomic across family)</span><select value={learningSplit} onChange={(event) => setLearningSplit(event.target.value)} disabled={learningSplit === 'train' || learningSplit === 'validation'}><option value="unassigned">Unassigned</option><option value="pilot_development">Pilot development</option>{learningSplit === 'train' ? <option value="train">Train (locked)</option> : null}{learningSplit === 'validation' ? <option value="validation">Validation (locked)</option> : null}</select></label>
            <p className="curation-note">Pilot assignment updates every asset and crop in the source family together. Test assignment is never available here.</p></> : null}
            <div className="checkbox-row"><label className="checkbox-field"><input type="checkbox" checked={personCountUnknown} onChange={(event) => { setPersonCountUnknown(event.target.checked); if (event.target.checked) setPersonCount(null) }} /><span>Person count is uncertain</span></label></div>
            <label><span>Person count</span><input type="number" min={1} step={1} value={personCount ?? ''} disabled={personCountUnknown} onChange={(event) => { const text = event.target.value; const value = Number(text); setPersonCountUnknown(false); setPersonCount(text === '' ? null : Number.isSafeInteger(value) && value >= 1 ? value : null) }} /><small>Pose matching separately disables above eight people.</small></label>
            <label><span>Safety</span><select value={sfw} onChange={(event) => setSfw(event.target.value as typeof sfw)}><option value="">Choose after review</option><option value="uncertain">Uncertain</option><option value="safe">Safe</option><option value="unsafe">Unsafe</option></select></label>
            <fieldset><legend>Line-art / reference quality</legend><div className="segmented-control">{[1, 2, 3].map((value) => <button key={value} type="button" className={quality === value ? 'is-active' : ''} aria-pressed={quality === value} onClick={() => setQuality(value)}>{value}</button>)}</div></fieldset>
            <label><span>Useful for character design?</span><select value={characterUseful} onChange={(event) => setCharacterUseful(event.target.value as typeof characterUseful)}><option value="">Choose after review</option><option value="yes">Yes</option><option value="no">No</option></select></label>
            <label><span>Text / balloon coverage (0–1)</span><input type="number" min={0} max={1} step={0.05} value={textCoverage ?? ''} onChange={(event) => { const value = Number(event.target.value); setTextCoverage(event.target.value === '' ? null : Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null) }} /></label>
            <div className="checkbox-row"><label className="checkbox-field"><input type="checkbox" checked={clipped} onChange={(event) => setClipped(event.target.checked)} /><span>Clipped</span></label><label className="checkbox-field"><input type="checkbox" checked={damaged} onChange={(event) => setDamaged(event.target.checked)} /><span>Damaged</span></label></div>
            <fieldset className="viewpoint-fields"><legend>Visible viewpoint · rubric v1</legend>
              <label className="checkbox-field"><input type="checkbox" checked={viewpointReviewed} disabled={viewpointNotApplicable} onChange={(event) => setViewpointReviewed(event.target.checked)} /><span>I reviewed the visible viewpoint</span></label>
              <label className="checkbox-field"><input type="checkbox" checked={viewpointNotApplicable} onChange={(event) => { setViewpointNotApplicable(event.target.checked); if (event.target.checked) setViewpointReviewed(false) }} /><span>Not applicable</span></label>
              {!viewpointNotApplicable ? <>
                <label><span>Evidence basis</span><select value={evidenceBasis} onChange={(event) => setEvidenceBasis(event.target.value as typeof evidenceBasis)}><option value="visible_crop">Visible crop</option><option value="crop_with_parent_context">Crop + parent context</option></select></label>
                {(Object.keys(AXIS_OPTIONS) as Axis[]).map((axis) => <label key={axis}><span>{axis.replaceAll('_', ' ')}</span><select value={axes[axis]} disabled={!viewpointReviewed} onChange={(event) => setAxes((current) => ({ ...current, [axis]: event.target.value }))}>{AXIS_OPTIONS[axis].map((value) => <option key={value} value={value}>{value.replaceAll('_', ' ')}</option>)}</select></label>)}
              </> : <p className="curation-note">A not-applicable decision is stored with your reviewer ID, time, and rationale.</p>}
            </fieldset>
            {viewpointNotApplicable ? <label><span>Why is viewpoint not applicable?</span><textarea value={viewpointReason} onChange={(event) => setViewpointReason(event.target.value)} minLength={10} maxLength={1000} /></label> : null}
            <label><span>Keep / reject rationale (required)</span><textarea value={reason} onChange={(event) => setReason(event.target.value)} required maxLength={2000} placeholder="Briefly explain your choice, e.g. clear head sketch or blurry image." /></label>
            <label className="note-field"><span>Review note</span><textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} placeholder="Optional context; do not include personal data." /></label>
            {reviewBlocker ? <p id="review-action-blocker" className="curation-note" role="status">{reviewBlocker}</p> : null}
            {!labelsComplete ? <p id="keep-label-blocker" className="curation-note" role="status">To keep this image, complete: {missingKeepLabels.join(', ')}. Reject does not require these labels.</p> : null}
            <div className="review-actions">
              <Button className="button--danger" disabled={Boolean(reviewBlocker)} aria-describedby={reviewBlocker ? 'review-action-blocker' : undefined} onClick={() => void submitReview('reject')}><X size={16} />Reject</Button>
              <Button className="button--primary" disabled={Boolean(reviewBlocker) || !labelsComplete} aria-describedby={[reviewBlocker ? 'review-action-blocker' : '', !labelsComplete ? 'keep-label-blocker' : ''].filter(Boolean).join(' ') || undefined} onClick={() => void submitReview('keep')}><Check size={16} />Keep</Button>
            </div>
          {fixedYieldMode ? <fieldset className="viewpoint-fields"><legend>M0 fixed-sample yield judgment</legend><p>Could this source produce a clear, useful eye/detail crop for same-viewpoint reference search?</p><label><span>Eye/detail crop usefulness</span><select value={usefulEyeCrop} onChange={(event) => setUsefulEyeCrop(event.target.value as EyeCropYield)}><option value="">Choose one</option><option value="yes">Yes</option><option value="no">No</option><option value="uncertain">Uncertain</option></select></label></fieldset> : null}
            <Button className="button--wide" disabled={Boolean(reviewBlocker)} aria-describedby={reviewBlocker ? 'review-action-blocker' : undefined} onClick={() => void submitReview('quarantine')}>Keep quarantined</Button>
          </> : <p className="curation-note">Select a cataloged item. Locked-test assets are never listed here.</p>}
          {error ? <p className="curation-error" role="alert">{error}</p> : null}
          <p className="curation-note">{prototypeMode ? 'Your image labels and detail crops are saved locally for the educational prototype.' : 'Reviews remain local. Rights are inherited from approved source records; this screen cannot grant permissions or activate a gallery.'}</p>
        </aside>
      </div>
    </ResearchShell>
  )
}
