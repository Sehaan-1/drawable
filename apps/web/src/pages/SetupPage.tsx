import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react'
import { AlertCircle, CheckCircle2, Cpu, Database, ExternalLink, FileCheck2, RefreshCw, ShieldCheck } from 'lucide-react'
import { ResearchShell } from '../components/AppChrome'
import { Button, StatusDot } from '../components/primitives'

type PermissionKey = 'local_display' | 'training' | 'evaluation' | 'derive' | 'trace' | 'export' | 'public_demo' | 'redistribute' | 'cloud_storage' | 'cloud_processing'
type Permissions = Record<PermissionKey, boolean>
type Work = { id: string; title: string; artist: string; license: string; decision: string; permissions: Permissions; attribution_text: string | null; selected: boolean }
type Source = { id: string; title: string; authority_url: string; terms_url: string; decision: string; terms_snapshot_sha256: string | null; permissions: Permissions; archive: { status?: string; byte_length?: number; error?: string } | null; candidate_count: number }
type Model = { id: string; title: string; terms_url: string; decision: string; terms_snapshot_sha256: string | null; weights_state: string; training_authorized: false }
type Preparation = { status: string; job_id?: string; phase?: string; message?: string; manifest_path?: string; catalogued_assets?: number; resumable?: boolean; assets_approved?: boolean; reconciliation_status?: string; clean_extraction_status?: string }
type Workspace = { schema_version: 1; revision: number; selected_sources: Source[]; selected_works: Work[]; available_works: Work[]; selected_models: Model[]; reviewer_id: string; preparation: Preparation; curation: { total: number; pending: number; candidate: number; rejected: number; reviewed: number }; required_handoff: string; training_authorized: false }
type DecisionKind = 'source' | 'work' | 'model'
type SavedTerms = { sha256: string; url: string; path: string }

const permissionLabels: { key: PermissionKey; label: string; note: string }[] = [
  { key: 'local_display', label: 'Show images in this local review', note: 'Allows local previews; it does not enable real search or a serving gallery.' },
  { key: 'evaluation', label: 'Use approved items in local retrieval evaluation', note: 'Does not train a model or share data.' },
  { key: 'derive', label: 'Create and store manually selected crops', note: 'Use only if the reviewed terms allow local derivatives.' },
  { key: 'trace', label: 'Allow the drawing app to trace an image', note: 'Separate from showing an image.' },
  { key: 'export', label: 'Allow an image to appear in a saved/exported project', note: 'Separate from local display.' },
  { key: 'training', label: 'Use images for model training', note: 'Separate approval; no training starts in this workflow.' },
  { key: 'public_demo', label: 'Use images in a public demonstration', note: 'Leave off unless the terms clearly allow it.' },
  { key: 'redistribute', label: 'Redistribute original or derived images', note: 'Separate from showing images in the app.' },
  { key: 'cloud_storage', label: 'Store images with a cloud provider', note: 'No transfer occurs from granting this permission.' },
  { key: 'cloud_processing', label: 'Process images with a cloud provider', note: 'Cloud storage must also be allowed.' },
]

const emptyPermissions = (): Permissions => ({ local_display: false, training: false, evaluation: false, derive: false, trace: false, export: false, public_demo: false, redistribute: false, cloud_storage: false, cloud_processing: false })

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, credentials: 'same-origin', cache: 'no-store' })
  const value = await response.json().catch(() => ({})) as { detail?: string }
  if (!response.ok) throw new Error(value.detail ?? `Local setup request failed (${response.status})`)
  return value as T
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

function readReviewerId() {
  try { return sessionStorage.getItem('drawable-curation-reviewer') ?? '' } catch { return '' }
}

function formatBytes(value?: number) {
  if (!value || value < 1) return 'not acquired'
  const gb = value / (1024 ** 3)
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.ceil(value / (1024 ** 2))} MB`
}

function currentDecision(value: string | undefined) {
  return value === 'approved' ? 'Approved' : value === 'rejected' ? 'Declined' : 'Needs your review'
}

function PermissionChoices({ value, onChange, disabled = false }: { value: Permissions; onChange: (next: Permissions) => void; disabled?: boolean }) {
  return <div className="review-permissions">{permissionLabels.map(({ key, label, note }) => <label className="review-permission" key={key}>
    <input type="checkbox" checked={value[key]} disabled={disabled} onChange={(event) => onChange({ ...value, [key]: event.target.checked })} />
    <span><strong>{label}</strong><small>{note}</small></span>
  </label>)}</div>
}

export default function SetupPage() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [reviewerId, setReviewerId] = useState(readReviewerId)
  const [terms, setTerms] = useState<Record<string, SavedTerms>>({})
  const [sourcePermissions, setSourcePermissions] = useState<Permissions>(emptyPermissions)
  const [sourcePermissionsTouched, setSourcePermissionsTouched] = useState(false)
  const [workPermissions, setWorkPermissions] = useState<Record<string, Permissions>>({})
  const [modelDecisions, setModelDecisions] = useState<Record<string, { decision: 'approved' | 'rejected'; note: string }>>({})
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [attributions, setAttributions] = useState<Record<string, string>>({})
  const [batchSize, setBatchSize] = useState(24)
  const [previewConfirmed, setPreviewConfirmed] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [jobId, setJobId] = useState('')
  const [downloadedBytes, setDownloadedBytes] = useState<number | null>(null)
  const [workSelection, setWorkSelection] = useState<string[]>([])

  const refresh = useCallback(async () => {
    const next = await api<Workspace>('/api/v1/setup')
    setWorkspace(next)
    setWorkSelection(next.selected_works.map((work) => work.id))
    if (!sourcePermissionsTouched && next.selected_sources[0]) setSourcePermissions(next.selected_sources[0].permissions)
    setWorkPermissions((current) => {
      const merged = { ...current }
      for (const work of next.available_works) if (!merged[work.id]) merged[work.id] = work.permissions
      return merged
    })
    if (next.reviewer_id) setReviewerId(next.reviewer_id)
    return next
  }, [sourcePermissionsTouched])

  useEffect(() => { void refresh().catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Local setup API is unavailable.')) }, [refresh])

  useEffect(() => {
    if (!jobId) return
    let closed = false
    const poll = async () => {
      try {
        const next = await api<{ status: string; message: string; progress: Record<string, unknown> }>(`/api/v1/setup/prepare-candidates/${encodeURIComponent(jobId)}`)
        if (closed) return
        setNotice(next.message)
        setDownloadedBytes(typeof next.progress.downloaded_bytes === 'number' ? next.progress.downloaded_bytes : null)
        await refresh()
        if (next.status === 'complete') {
          setJobId('')
          setNotice('Candidates are ready. Grant local previews for a small batch when you are ready to review images.')
        } else if (next.status === 'blocked') {
          setJobId('')
          setError(next.message)
        }
      } catch (reason) {
        if (!closed) setError(reason instanceof Error ? reason.message : 'Could not read preparation progress.')
      }
    }
    const timer = window.setInterval(() => { void poll() }, 2000)
    void poll()
    return () => { closed = true; window.clearInterval(timer) }
  }, [jobId, refresh])

  const selectedWorkIds = useMemo(() => workSelection, [workSelection])
  const source = workspace?.selected_sources[0]
  const selectedReady = Boolean(source?.decision === 'approved' && source.permissions.local_display && workspace?.selected_works.some((work) => work.decision === 'approved' && work.permissions.local_display))
  const candidatesReady = workspace?.preparation.status === 'candidates_catalogued_for_preview'
  const needsIdentity = !reviewerId.trim()

  function setDecisionNote(id: string, value: string) { setNotes((current) => ({ ...current, [id]: value })) }

  async function capture(kind: 'source' | 'model', id: string) {
    setBusy(`capture:${kind}:${id}`); setError(''); setNotice('Capturing the pinned official terms page; this can take a moment.')
    try {
      const result = await api<{ terms_snapshot_sha256: string; source_url: string; final_url: string; snapshot_path: string }>(
        '/api/v1/setup/terms', json('POST', { kind, id }),
      )
      setTerms((current) => ({ ...current, [`${kind}:${id}`]: { sha256: result.terms_snapshot_sha256, url: result.source_url || result.final_url, path: result.snapshot_path } }))
      await refresh()
      setNotice('Terms copy saved privately. Open the official page, review the terms, then record your own decision.')
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not capture terms.') }
    finally { setBusy('') }
  }

  async function recordDecision(kind: DecisionKind, id: string, decision: 'approved' | 'rejected', permissions?: Permissions, attribution?: string, noteOverride?: string) {
    if (!workspace) return
    const key = `${kind}:${id}`
    const captured = terms[key]
    if (!reviewerId.trim()) { setError('Enter a reviewer ID first.'); return }
    if (!captured?.sha256) { setError('Capture the current official terms copy before deciding.'); return }
    const note = noteOverride ?? notes[key] ?? ''
    if (note.trim().length < 12) { setError('Add a short note about the terms you reviewed.'); return }
    setBusy(`decision:${key}`); setError(''); setNotice('')
    try {
      await api('/api/v1/setup/decisions', json('POST', {
        expected_revision: workspace.revision, kind, id, reviewer_id: reviewerId.trim(), decision,
        terms_snapshot_sha256: captured.sha256, note,
        permissions: kind === 'model' ? null : permissions ?? emptyPermissions(),
        attribution_text: kind === 'work' ? (attribution ?? '') : null,
      }))
      if (kind === 'source') {
        setSourcePermissions(permissions ?? emptyPermissions())
        setSourcePermissionsTouched(false)
      }
      if (kind === 'work') setWorkPermissions((current) => ({ ...current, [id]: permissions ?? emptyPermissions() }))
      sessionStorage.setItem('drawable-curation-reviewer', reviewerId.trim())
      setNotice(`${kind === 'work' ? workspace.available_works.find((work) => work.id === id)?.title ?? 'Work' : kind === 'model' ? workspace.selected_models.find((model) => model.id === id)?.title ?? 'Model' : source?.title ?? 'Source'} decision saved privately.`)
      await refresh()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not save the review decision.') }
    finally { setBusy('') }
  }

  async function saveWorks(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!workspace) return
    const ids = workSelection
    if (!ids.length) { setError('Choose at least one work.'); return }
    setBusy('works'); setError('')
    try {
      await api('/api/v1/setup/works', json('PUT', { expected_revision: workspace.revision, work_ids: ids }))
      await refresh(); setNotice('Selected works saved. Only these work decisions can open the local review queue.')
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not save work selection.') }
    finally { setBusy('') }
  }

  async function prepareCandidates() {
    if (!workspace) return
    if (!reviewerId.trim()) { setError('Enter a reviewer ID first.'); return }
    setBusy('prepare'); setError(''); setNotice('')
    try {
      const next = await api<{ job_id: string }>('/api/v1/setup/prepare-candidates', json('POST', {
        expected_revision: workspace.revision, reviewer_id: reviewerId.trim(),
      }))
      sessionStorage.setItem('drawable-curation-reviewer', reviewerId.trim())
      setDownloadedBytes(null); setJobId(next.job_id); setNotice('Checking the source archive, then fingerprinting images into the private review queue.')
      await refresh()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not prepare review candidates.') }
    finally { setBusy('') }
  }

  async function grantPreviewBatch() {
    if (!workspace) return
    if (!previewConfirmed) { setError('Confirm that this batch may be shown locally before preparing previews.'); return }
    setBusy('preview'); setError('')
    try {
      const result = await api<{ count: number; review_url: string }>('/api/v1/setup/preview-batches', json('POST', {
        expected_revision: workspace.revision, reviewer_id: reviewerId.trim(), limit: batchSize, confirm_local_preview: true,
      }))
      setPreviewConfirmed(false)
      setNotice(`${result.count} images are available for review. They are still unlabelled and are not in a gallery.`)
      await refresh()
      window.location.assign(result.review_url)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not prepare local previews.') }
    finally { setBusy('') }
  }

  if (!workspace) return <ResearchShell eyebrow="Local setup" title="Source and model review"><div className="setup-page"><div className="review-banner"><AlertCircle size={18} />{error || 'Connecting to the private local review service…'}</div></div></ResearchShell>

  return <ResearchShell eyebrow="Local research setup" title="Prepare a real reference review">
    <div className="setup-page review-setup">
      <section className="review-setup-hero">
        <div className="review-setup-icon"><ShieldCheck size={24} /></div>
        <div><span className="dock-eyebrow">Private local workspace</span><h1>Review source terms, then review images</h1><p>This first step records only the uses you approve. Model weights stay unloaded, cloud permissions are off by default, and an image stays out of the gallery until you label it.</p></div>
        <Button onClick={() => { setError(''); void refresh().catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Could not refresh.')) }}><RefreshCw size={15} />Refresh</Button>
      </section>

      {error ? <div className="review-banner review-banner--error" role="alert"><AlertCircle size={18} />{error}</div> : null}
      {notice ? <div className="review-banner" role="status"><CheckCircle2 size={18} />{notice}</div> : null}

      <section className="review-step-card">
        <header className="review-step-heading"><span>1</span><div><h2>Identify the reviewer</h2><p>Use a private nickname or ID. It is saved with decisions for your own audit trail.</p></div></header>
        <label className="review-input"><span>Reviewer ID</span><input value={reviewerId} autoComplete="nickname" onChange={(event) => setReviewerId(event.target.value)} placeholder="For example, owner-local" maxLength={80} /></label>
      </section>

      <section className="review-step-card">
        <header className="review-step-heading"><span>2</span><div><h2>Review the source terms</h2><p>Open the official page, then capture a copy. Capturing terms does not approve any use.</p></div></header>
        {workspace.selected_sources.map((item) => {
          const key = `source:${item.id}`
          const captureState = terms[key]
          const saved = item.terms_snapshot_sha256
          const needsDecision = !captureState && saved
          return <article className="review-owner-card" key={item.id}>
            <div className="review-owner-card__top"><div><h3>{item.title}</h3><p>{item.candidate_count.toLocaleString()} local catalog candidates · {item.archive?.status ?? 'archive not yet verified'} · {formatBytes(item.archive?.byte_length)}</p></div><span className={`review-state ${item.decision === 'approved' ? 'is-approved' : item.decision === 'rejected' ? 'is-declined' : ''}`}>{currentDecision(item.decision)}</span></div>
            <div className="review-link-row"><a href={item.terms_url} target="_blank" rel="noopener noreferrer">Open official source page <ExternalLink size={13} /></a><Button disabled={Boolean(busy) || needsIdentity} onClick={() => void capture('source', item.id)}><FileCheck2 size={14} />{busy === `capture:source:${item.id}` ? 'Capturing…' : 'Capture current terms'}</Button></div>
            {captureState || saved ? <p className="review-hash">Terms copy: <code>{captureState?.sha256 ?? saved}</code> {captureState ? `· saved privately as ${captureState.path}` : '· previously captured; capture again before a new decision'}</p> : null}
            {captureState ? <details className="review-terms-document"><summary>Read the exact captured terms copy</summary><iframe sandbox="" title={`Captured terms for ${item.title}`} src={`/api/v1/setup/terms/source/${encodeURIComponent(item.id)}`} /></details> : null}
            <div className="review-owner-form">
              <div className="review-form-title"><strong>Decide which uses the source permits</strong><small>Every choice starts off. Keep training, cloud, and public-demo uses off unless you separately approve them.</small></div>
              <PermissionChoices value={sourcePermissions} onChange={(value) => { setSourcePermissions(value); setSourcePermissionsTouched(true) }} />
              <label className="review-input"><span>What did you check?</span><textarea value={notes[key] ?? ''} onChange={(event) => setDecisionNote(key, event.target.value)} maxLength={1000} placeholder="Name the terms page/version and the uses you checked." /></label>
              <div className="review-action-row"><Button className="review-accept" disabled={Boolean(busy) || !captureState?.sha256 || needsIdentity} onClick={() => void recordDecision('source', item.id, 'approved', sourcePermissions)}><CheckCircle2 size={14} />{item.decision === 'approved' ? 'Update source permissions' : 'Record approval'}</Button><Button className="review-decline" disabled={Boolean(busy) || !captureState?.sha256 || needsIdentity} onClick={() => void recordDecision('source', item.id, 'rejected', emptyPermissions())}>{item.decision === 'approved' ? 'Revoke source approval' : 'Decline this source'}</Button></div>
            </div>
          </article>
        })}
      </section>

      <section className="review-step-card">
        <header className="review-step-heading"><span>3</span><div><h2>Select the works to review</h2><p>These are source groupings, not approved uses. Selection only narrows the private catalog.</p></div></header>
        <form onSubmit={(event) => void saveWorks(event)}>
          <div className="review-work-grid">{workspace.available_works.map((work) => <label className="review-work-option" key={work.id}><input type="checkbox" name="work" value={work.id} checked={selectedWorkIds.includes(work.id)} onChange={(event) => setWorkSelection((current) => event.target.checked ? [...current, work.id] : current.filter((id) => id !== work.id))} /><span><strong>{work.title}</strong><small>{work.artist} · {work.license}</small><em>{currentDecision(work.decision)}</em></span></label>)}</div>
          <div className="review-action-row"><Button disabled={Boolean(busy)}><Database size={14} />{busy === 'works' ? 'Saving…' : 'Save selected works'}</Button></div>
        </form>
      </section>

      <section className="review-step-card">
        <header className="review-step-heading"><span>4</span><div><h2>Review each selected work</h2><p>A work needs its own decision. Local display does not automatically permit crops, tracing, exports, model training, or public demos.</p></div></header>
        {!workspace.selected_sources.some((item) => item.decision === 'approved') ? <div className="review-inline-note">Approve the source terms first. The work forms will then let you record permission for each selected work.</div> : null}
        {workspace.selected_works.map((work) => {
          const key = `work:${work.id}`
          const captureState = terms[`source:${source?.id}`]
          const selectedPermissions = workPermissions[work.id] ?? emptyPermissions()
          const savedChoice = work.decision === 'approved'
          return <article className="review-owner-card review-owner-card--work" key={work.id}>
            <div className="review-owner-card__top"><div><h3>{work.title}</h3><p>{work.artist} · {work.license}</p></div><span className={`review-state ${savedChoice ? 'is-approved' : work.decision === 'rejected' ? 'is-declined' : ''}`}>{currentDecision(work.decision)}</span></div>
            <a className="review-inline-link" href={source?.authority_url} target="_blank" rel="noopener noreferrer">Review this work on the official source page <ExternalLink size={13} /></a>
            <label className="review-input"><span>Attribution text from the source</span><input value={attributions[work.id] ?? work.attribution_text ?? work.artist} onChange={(event) => setAttributions((current) => ({ ...current, [work.id]: event.target.value }))} maxLength={1000} /></label>
            <PermissionChoices value={selectedPermissions} onChange={(value) => setWorkPermissions((current) => ({ ...current, [work.id]: value }))} />
            <label className="review-input"><span>What did you check?</span><textarea value={notes[key] ?? ''} onChange={(event) => setDecisionNote(key, event.target.value)} maxLength={1000} placeholder="Record the work and use permissions you verified." /></label>
            <div className="review-action-row"><Button className="review-accept" disabled={Boolean(busy) || !captureState?.sha256 || !workspace.selected_sources.some((item) => item.decision === 'approved') || needsIdentity} onClick={() => void recordDecision('work', work.id, 'approved', selectedPermissions, attributions[work.id] ?? work.attribution_text ?? work.artist)}><CheckCircle2 size={14} />{savedChoice ? 'Update work permissions' : 'Record work approval'}</Button><Button className="review-decline" disabled={Boolean(busy) || !captureState?.sha256 || needsIdentity} onClick={() => void recordDecision('work', work.id, 'rejected', emptyPermissions())}>{savedChoice ? 'Revoke work approval' : 'Decline this work'}</Button></div>
          </article>
        })}
      </section>

      <section className="review-step-card">
        <header className="review-step-heading"><span>5</span><div><h2>Prepare a small image-review batch</h2><p>This checks the official archive and catalogs only works you approved for local display. The download can take a while; progress and partial-download recovery appear here. Preview batches contain native sketches.</p></div></header>
        {workspace.preparation.status === 'blocked' ? <div className="review-inline-note review-inline-note--error">{workspace.preparation.message} {workspace.preparation.resumable ? 'Press prepare again to resume safely.' : ''}</div> : null}
        {workspace.preparation.status === 'queued' || workspace.preparation.status === 'running' ? <div className="review-inline-note"><RefreshCw size={15} className="review-spin" />{workspace.preparation.message ?? 'Preparing your review batch…'}{workspace.preparation.phase ? ` (${workspace.preparation.phase}${downloadedBytes ? ` · ${formatBytes(downloadedBytes)} downloaded` : ''})` : ''}</div> : null}
        {workspace.preparation.status === 'candidates_catalogued_for_preview' ? <div className="review-inline-note"><CheckCircle2 size={15} />Private catalog ready: {workspace.preparation.catalogued_assets?.toLocaleString?.() ?? workspace.curation.total.toLocaleString()} records. The clean official extraction is verified; {workspace.preparation.reconciliation_status === 'reconciled' ? 'the older local tree also matches.' : 'the older local tree differs and was left untouched.'} Images are still unreviewed.</div> : null}
        <div className="review-batch-controls"><label className="review-input"><span>Batch size (up to 32)</span><select value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))}>{[8, 16, 24, 32].map((size) => <option value={size} key={size}>{size} candidates</option>)}</select></label>
          {candidatesReady ? <label className="review-consent"><input type="checkbox" checked={previewConfirmed} onChange={(event) => setPreviewConfirmed(event.target.checked)} /><span>I confirm that the selected, specifically approved works may be shown to me locally for reference review.</span></label> : null}
          <div className="review-action-row">{candidatesReady ? <Button className="review-accept" disabled={Boolean(busy) || !selectedReady || needsIdentity || !previewConfirmed} onClick={() => void grantPreviewBatch()}><Database size={14} />{busy === 'preview' ? 'Preparing images…' : 'Prepare local previews'}</Button> : <Button className="review-accept" disabled={Boolean(busy) || !selectedReady || needsIdentity || Boolean(jobId)} onClick={() => void prepareCandidates()}><FileCheck2 size={14} />{busy === 'prepare' || jobId ? 'Preparing…' : 'Prepare candidates'}</Button>}</div>
        </div>
        {workspace.preparation.status === 'preview_batch_ready' ? <a className="review-ready-link" href="/curate">Open the curation review workspace <ExternalLink size={14} /></a> : null}
      </section>

      <section className="review-step-card review-step-card--models">
        <header className="review-step-heading"><span><Cpu size={17} /></span><div><h2>Model terms (can wait until after image review)</h2><p>Inference terms are separate from training terms. This decision never loads weights or authorizes fine-tuning.</p></div></header>
        {workspace.selected_models.map((model) => {
          const key = `model:${model.id}`
          const captureState = terms[key]
          const choice = modelDecisions[model.id] ?? { decision: 'approved' as const, note: '' }
          return <article className="review-model-row" key={model.id}><div className="review-owner-card__top"><div><h3>{model.title}</h3><p>Weights: {model.weights_state} · inference only</p></div><span className={`review-state ${model.decision === 'approved' ? 'is-approved' : model.decision === 'rejected' ? 'is-declined' : ''}`}>{currentDecision(model.decision)}</span></div>
            <div className="review-link-row"><a href={model.terms_url} target="_blank" rel="noopener noreferrer">Open pinned official model page <ExternalLink size={13} /></a><Button disabled={Boolean(busy) || needsIdentity} onClick={() => void capture('model', model.id)}><FileCheck2 size={14} />Capture model terms</Button></div>
            {captureState ? <p className="review-hash">Pinned terms copy: <code>{captureState.sha256}</code></p> : null}
            {captureState ? <details className="review-terms-document"><summary>Read the exact captured model terms copy</summary><iframe sandbox="" title={`Captured inference terms for ${model.title}`} src={`/api/v1/setup/terms/model/${encodeURIComponent(model.id)}`} /></details> : null}
            <label className="review-input"><span>Terms decision</span><select value={choice.decision} onChange={(event) => setModelDecisions((current) => ({ ...current, [model.id]: { ...choice, decision: event.target.value as 'approved' | 'rejected' } }))}><option value="approved">I approve local inference</option><option value="rejected">I decline local inference</option></select></label>
            <label className="review-input"><span>What did you check?</span><textarea value={choice.note} onChange={(event) => setModelDecisions((current) => ({ ...current, [model.id]: { ...choice, note: event.target.value } }))} maxLength={1000} placeholder="Mention the pinned model page and inference-use terms." /></label>
            <div className="review-action-row"><Button disabled={Boolean(busy) || !captureState?.sha256 || needsIdentity || choice.note.trim().length < 12} onClick={() => void recordDecision('model', model.id, choice.decision, undefined, undefined, choice.note)}><CheckCircle2 size={14} />Save model decision</Button></div>
          </article>
        })}
      </section>

      <section className="review-progress-card"><div><span className="dock-eyebrow">Current state</span><strong>{workspace.curation.reviewed.toLocaleString()} reviewed · {workspace.curation.candidate.toLocaleString()} labeled keep · {workspace.curation.rejected.toLocaleString()} rejected</strong><small>Model weights are {workspace.selected_models.every((model) => model.weights_state === 'complete') ? 'locally present' : 'not yet acquired'}; training is not authorized.</small></div><StatusDot tone={candidatesReady ? 'success' : 'warning'} /></section>

      <footer className="review-setup-footer"><p><ShieldCheck size={15} />Decisions and source files stay in the private local data folder. Every preview requires current source and work permissions plus your explicit preview confirmation.</p><p><AlertCircle size={15} />An image is still only a candidate until you review its labels. This workflow does not activate a serving gallery.</p></footer>
    </div>
  </ResearchShell>
}
