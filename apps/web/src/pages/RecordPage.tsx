import { useEffect, useRef, useState } from 'react'
import type { StoreApi } from 'zustand/vanilla'
import { Link } from 'react-router-dom'
import type { StrokeOperation } from '../lib/types'
import { CanvasStage } from '../components/CanvasStage'
import { ToolRail } from '../components/ToolRail'
import { DocumentStoreProvider, createDocumentStore, type DocumentCommand, type DocumentState } from '../state/documentStore'
import { countDocument, validateAndReplay } from '../services/recordingValidation'
import { deleteAcceptedRecording, exportAcceptedRecordingBatch, listAcceptedRecordings, saveAcceptedRecording } from '../services/recordingPersistence'
import { checksumRenderedDocument } from '../services/renderChecksum'
import {
  RECORDING_FORMAT, RECORDING_FORMAT_VERSION, RECORDING_MIME, RECORDING_EXTENSION,
  RECORDING_SCOPES, RECORDING_STYLES, type DrawingSequenceCurrent, type RecordingEventV2,
  type RecordingMutation, type RecordingReferenceKind, type RecordingRightsBasis,
  type RecordingScope, type RecordingStyle, type RecordingReferencePermissions,
} from '../services/recordingTypes'

type Phase = 'intro' | 'consent' | 'ready' | 'recording' | 'review' | 'accepted'

function newId(prefix: string) {
  return `${prefix}-${crypto.randomUUID()}`
}

function stableLayerId(value: string | undefined): RecordingEventV2['activeLayerId'] {
  return value === 'layer-2' || value === 'layer-3' || value === 'layer-4' ? value : 'layer-1'
}

function normalizeCommand(command: DocumentCommand, atMs: number): RecordingMutation | null {
  if (command.type !== 'operation_committed' && command.type !== 'layer_updated' && command.type !== 'layer_moved' && command.type !== 'layer_cleared' && command.type !== 'undo' && command.type !== 'redo') return null
  if (command.type !== 'operation_committed') return structuredClone(command)
  const operation = structuredClone(command.operation)
  const firstTime = operation.points[0]?.time ?? 0
  operation.points = operation.points.map((point) => ({ ...point, time: Math.max(0, atMs + point.time - firstTime) }))
  operation.createdAt = atMs
  return { ...command, operation }
}

export default function RecordPage() {
  const [phase, setPhase] = useState<Phase>('intro')
  const [scope, setScope] = useState<RecordingScope>('face_head')
  const [style, setStyle] = useState<RecordingStyle>('manga_anime')
  const [taskId, setTaskId] = useState('pilot-character-01')
  const [taskInstructions, setTaskInstructions] = useState('Draw the assigned character study using the requested style.')
  const [collectionProgram, setCollectionProgram] = useState<'pilot-development' | 'full-program'>('pilot-development')
  const [contributorGroupId, setContributorGroupId] = useState('local-pilot')
  const [referenceKind, setReferenceKind] = useState<RecordingReferenceKind>('none')
  const [referenceAssetId, setReferenceAssetId] = useState('')
  const [referenceUrl, setReferenceUrl] = useState('')
  const [referenceTitle, setReferenceTitle] = useState('')
  const [referenceRights, setReferenceRights] = useState<RecordingRightsBasis>('owned')
  const [referenceLicense, setReferenceLicense] = useState('')
  const [referenceAttribution, setReferenceAttribution] = useState('')
  const [referencePermittedUses, setReferencePermittedUses] = useState<RecordingReferencePermissions>({
    localDisplay: false, training: false, derivation: false, traceViewing: false,
    export: false, publicDemonstration: false, redistribution: false,
    cloudStorage: false, cloudProcessing: false,
  })
  const [research, setResearch] = useState(false)
  const [evaluation, setEvaluation] = useState(false)
  const [training, setTraining] = useState(false)
  const [publicDemo, setPublicDemo] = useState(false)
  const [cloudStorage, setCloudStorage] = useState(false)
  const [cloudProcessing, setCloudProcessing] = useState(false)
  const [creatorConfirmed, setCreatorConfirmed] = useState(false)
  const [canGrantPermissions, setCanGrantPermissions] = useState(false)
  const [referenceRightsDeclared, setReferenceRightsDeclared] = useState(false)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [accepted, setAccepted] = useState<DrawingSequenceCurrent[]>([])
  const storeRef = useRef<StoreApi<DocumentState> | null>(null)
  const eventsRef = useRef<RecordingEventV2[]>([])
  const renderedChecksumRef = useRef('')
  const recordingRef = useRef<{ id: string; createdAt: string; finishedAt: string | null; startedAt: number } | null>(null)

  const ensureStore = () => {
    if (!storeRef.current) {
      storeRef.current = createDocumentStore({
        onCommand: (command) => {
          const recording = recordingRef.current
          if (!recording) return
          const atMs = Math.max(0, Math.round(performance.now() - recording.startedAt))
          const normalized = normalizeCommand(command, atMs)
          if (!normalized) return
          eventsRef.current.push({
            eventId: crypto.randomUUID(),
            seq: eventsRef.current.length + 1,
            atMs,
            documentRevision: (storeRef.current?.getState().document.revision ?? 0) + 1,
            activeLayerId: stableLayerId(storeRef.current?.getState().activeLayerId),
            action: { type: 'document_mutation', command: normalized },
          })
        },
      })
    }
    return storeRef.current
  }

  useEffect(() => {
    if (phase !== 'recording') return
    const timer = window.setInterval(() => {
      if (recordingRef.current) setElapsed(Math.round(performance.now() - recordingRef.current.startedAt))
    }, 100)
    const beforeUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', beforeUnload)
    return () => { window.clearInterval(timer); window.removeEventListener('beforeunload', beforeUnload) }
  }, [phase])

  const startRecording = () => {
    if (!research || !evaluation || !creatorConfirmed || !canGrantPermissions || !referenceRightsDeclared) {
      setError('Research, evaluation, creator, and reference-rights confirmations are required before recording.')
      setPhase('consent')
      return
    }
    const store = ensureStore()
    store.getState().newDocument()
    eventsRef.current = []
    renderedChecksumRef.current = ''
    const now = new Date().toISOString()
    recordingRef.current = { id: newId('recording'), createdAt: now, finishedAt: null, startedAt: performance.now() }
    eventsRef.current.push({ eventId: crypto.randomUUID(), seq: 1, atMs: 0, documentRevision: 0, activeLayerId: stableLayerId(store.getState().activeLayerId), action: { type: 'recording_started' } })
    setElapsed(0)
    setError('')
    setPhase('recording')
  }

  const buildFinalDocument = () => {
    const store = storeRef.current
    if (!store) return null
    const document = structuredClone(store.getState().document)
    document.trace = { ...document.trace, assetId: null, imageUrl: null }
    const normalizedOperations = new Map<string, StrokeOperation>()
    for (const event of eventsRef.current) {
      if (event.action.type === 'document_mutation' && event.action.command.type === 'operation_committed') {
        normalizedOperations.set(event.action.command.operation.id, event.action.command.operation)
      }
    }
    for (const layer of document.layers) {
      layer.operations = layer.operations.map((operation) => normalizedOperations.get(operation.id) ?? operation)
    }
    return document
  }

  const finishRecording = async () => {
    const store = storeRef.current
    if (!store || !recordingRef.current || recordingRef.current.finishedAt) return
    const hasStroke = store.getState().document.layers.some((layer) => layer.operations.some((operation) => operation.kind === 'stroke' && operation.tool !== 'eraser'))
    if (!hasStroke) { setError('Draw at least one visible stroke before finishing.'); return }
    const elapsedMs = Math.round(performance.now() - recordingRef.current.startedAt)
    try {
      const document = buildFinalDocument()
      if (!document) return
      renderedChecksumRef.current = await checksumRenderedDocument(document)
      recordingRef.current.finishedAt = new Date().toISOString()
      eventsRef.current.push({
        eventId: crypto.randomUUID(),
        seq: eventsRef.current.length + 1,
        atMs: elapsedMs,
        documentRevision: store.getState().document.revision,
        activeLayerId: stableLayerId(store.getState().activeLayerId),
        action: { type: 'recording_finished' },
      })
      setElapsed(elapsedMs)
      setError('')
      setPhase('review')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not verify the final rendered drawing.')
    }
  }

  const makeBundle = (): DrawingSequenceCurrent | null => {
    const recording = recordingRef.current
    if (!recording || !recording.finishedAt || !renderedChecksumRef.current) return null
    const document = buildFinalDocument()
    if (!document) return null
    return {
      format: RECORDING_FORMAT,
      formatVersion: RECORDING_FORMAT_VERSION,
      applicationVersion: '0.1.0',
      rendererVersion: 'canvas2d-v1',
      recordingId: recording.id,
      contributorGroupId: contributorGroupId.trim(),
      createdAt: recording.createdAt,
      completedAt: recording.finishedAt,
      acceptedAt: null,
      elapsedMs: elapsed,
      task: { taskId: taskId.trim() || 'pilot-character', instructions: taskInstructions.trim(), intendedScope: scope, intendedStyle: style, split: 'pilot_development', collectionProgram },
      consent: { consentVersion: 'research-v2', acceptedAt: recording.createdAt, creatorConfirmed, contributorCanGrantPermissions: canGrantPermissions, referenceRightsDeclared, research, training, evaluation, publicDemo, cloudStorage, cloudProcessing, retention: 'project-retained-until-withdrawal' },
      reference: referenceKind === 'none' ? { kind: 'none' } : {
        kind: referenceKind,
        ...(referenceKind === 'assigned-gallery' ? { assetId: referenceAssetId.trim() } : { sourceUrl: referenceUrl.trim() }),
        ...(referenceTitle.trim() ? { title: referenceTitle.trim() } : {}),
        rightsBasis: referenceRights,
        ...(referenceLicense.trim() ? { licenseId: referenceLicense.trim() } : {}),
        ...(referenceAttribution.trim() ? { attribution: referenceAttribution.trim() } : {}),
        permittedUses: { ...referencePermittedUses },
      },
      activeLayerId: stableLayerId(storeRef.current?.getState().activeLayerId),
      events: structuredClone(eventsRef.current),
      document,
      finalRenderedSha256: renderedChecksumRef.current,
      labelReview: { status: 'pending' },
      statistics: countDocument(document),
    }
  }

  const acceptRecording = async () => {
    const bundle = makeBundle()
    if (!bundle) return
    bundle.acceptedAt = new Date().toISOString()
    const validation = validateAndReplay(bundle)
    if (!validation.valid) { setError(validation.errors.join(' ')); return }
    try {
      await saveAcceptedRecording(bundle)
      setAccepted((current) => [...current, bundle])
      setPhase('accepted')
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save the accepted recording.')
    }
  }

  const discardRecording = () => {
    recordingRef.current = null
    eventsRef.current = []
    renderedChecksumRef.current = ''
    storeRef.current = null
    setError('')
    setPhase('intro')
  }

  const exportBundle = () => {
    const bundle = makeBundle()
    if (!bundle) return
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: RECORDING_MIME })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${bundle.recordingId}${RECORDING_EXTENSION}`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const exportAcceptedBatch = async () => {
    try {
      const blob = await exportAcceptedRecordingBatch()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `drawable-recordings-${new Date().toISOString().slice(0, 10)}.json`
      link.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not export the complete local recording batch.')
    }
  }

  const loadAccepted = async () => {
    try { setAccepted(await listAcceptedRecordings()) } catch { setError('Research recording storage is unavailable in this browser.') }
  }

  if ((phase === 'recording' || phase === 'review') && storeRef.current) {
    return (
      <DocumentStoreProvider store={storeRef.current}>
        <div className="draw-app">
          <header className="app-bar"><div className="workspace-switcher"><span className="wordmark">drawable</span><span className="workspace-name">Record</span></div><div className="document-cluster"><strong>{phase === 'recording' ? 'Consent-aware pilot recording' : 'Review recording'}</strong><span className="document-meta">{Math.floor(elapsed / 1000)}s · {scope} · {style}</span></div><div className="app-actions">{phase === 'recording' ? <><span className="save-status" role="status">Recording locally</span><button className="button button--primary" onClick={finishRecording}>Finish recording</button></> : <><span className="save-status" role="status">Review before acceptance</span><button className="button button--primary" onClick={acceptRecording}>Accept and save locally</button><button className="button" onClick={discardRecording}>Discard</button></>}</div></header>
          {phase === 'recording' ? <ToolRail /> : null}
          <main className="drawing-workspace"><CanvasStage searchIntegration="disabled" readOnly={phase === 'review'} /></main>
          {phase === 'review' ? <aside className="research-panel"><h2>Review your recording</h2><p>Elapsed {Math.round(elapsed / 1000)}s · {scope} · {style}</p><p>Task: {taskId}. {taskInstructions}</p><p>Reference: {referenceKind === 'none' ? 'none' : referenceKind === 'assigned-gallery' ? `${referenceAssetId || 'missing asset ID'} · ${referenceRights}` : `${referenceUrl || 'missing URL'} · ${referenceRights}`}</p><p>Consented: research {research ? 'yes' : 'no'}, evaluation {evaluation ? 'yes' : 'no'}, training reuse {training ? 'yes' : 'no'}, public demo {publicDemo ? 'yes' : 'no'}, cloud storage {cloudStorage ? 'yes' : 'no'}, cloud processing {cloudProcessing ? 'yes' : 'no'}.</p><p>Final active layer: {stableLayerId(storeRef.current?.getState().activeLayerId)} · rendered RGBA SHA-256: <code>{renderedChecksumRef.current}</code></p><p>The canvas is read-only. Accepted sequences are replay-validated and the final rendering is re-hashed before an atomic browser-local write. Export remains a local action.</p><button className="button" onClick={exportBundle}>Export sequence JSON</button>{error ? <p role="alert">{error}</p> : null}</aside> : null}
        </div>
      </DocumentStoreProvider>
    )
  }

  return (
    <main className="research-shell research-shell--record" data-testid="record-workspace">
      <header className="research-topbar"><Link className="workspace-switcher" to="/draw"><span className="wordmark">drawable</span><span className="workspace-name">Record</span></Link><div className="research-heading"><span>M0 feasibility</span><strong>Consent-aware drawing recorder</strong></div><span className="fixture-badge">Browser-local</span></header>
      <section className="research-main record-panels">
        <div className="research-panel">
          <span className="dock-eyebrow">Sequence recorder</span>
          <h1>{phase === 'review' ? 'Review your recording' : phase === 'accepted' ? 'Recording accepted' : 'Collect pilot evidence safely'}</h1>
          {phase === 'intro' ? <>
            <p>This workspace records editable vector strokes locally for the pilot. It does not search, upload, or add artwork to the gallery.</p>
            <button className="button button--primary" onClick={() => setPhase('consent')}>Configure a recording</button>
            <button className="button" onClick={loadAccepted}>Load accepted recordings</button>
            <button className="button" onClick={exportAcceptedBatch}>Export all accepted recordings</button>
          </> : null}
          {phase === 'consent' ? <>
            <label>Task ID<input maxLength={100} value={taskId} onChange={(event) => setTaskId(event.target.value)} /></label>
            <label>Task instructions<textarea maxLength={1000} rows={3} value={taskInstructions} onChange={(event) => setTaskInstructions(event.target.value)} /></label>
            <label>Collection program<select value={collectionProgram} onChange={(event) => setCollectionProgram(event.target.value as typeof collectionProgram)}><option value="pilot-development">Pilot development</option><option value="full-program">Full-program collection</option></select><small>This records why the sequence was collected. The locked train, model-selection, calibration, and test assignment still comes from a separate reviewed split manifest.</small></label>
            <label>Pseudonymous contributor-group code<input maxLength={100} value={contributorGroupId} onChange={(event) => setContributorGroupId(event.target.value)} /><small>Use a study code, not a real name, email, or other direct identifier. Keep the codebook separately.</small></label>
            <label>Intended scope<select value={scope} onChange={(event) => setScope(event.target.value as RecordingScope)}>{RECORDING_SCOPES.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <label>Intended style<select value={style} onChange={(event) => setStyle(event.target.value as RecordingStyle)}>{RECORDING_STYLES.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <fieldset className="record-fieldset"><legend>Reference provenance</legend>
              <label>Reference used<select value={referenceKind} onChange={(event) => setReferenceKind(event.target.value as RecordingReferenceKind)}><option value="none">No external or gallery reference</option><option value="assigned-gallery">Assigned gallery asset</option><option value="external">External reference</option></select></label>
              {referenceKind === 'assigned-gallery' ? <label>Source asset ID<input maxLength={200} value={referenceAssetId} onChange={(event) => setReferenceAssetId(event.target.value)} /></label> : null}
              {referenceKind === 'external' ? <label>Source URL (HTTPS)<input type="url" maxLength={2048} value={referenceUrl} onChange={(event) => setReferenceUrl(event.target.value)} /></label> : null}
              {referenceKind !== 'none' ? <>
                <label>Reference title<input value={referenceTitle} onChange={(event) => setReferenceTitle(event.target.value)} /></label>
                <label>Rights basis<select value={referenceRights} onChange={(event) => setReferenceRights(event.target.value as RecordingRightsBasis)}><option value="owned">Creator-owned</option><option value="licensed">Licensed</option><option value="public-domain">Public domain (verify source record)</option></select></label>
                <label>License or rights-record ID<input value={referenceLicense} onChange={(event) => setReferenceLicense(event.target.value)} /></label>
                <label>Required attribution<input value={referenceAttribution} onChange={(event) => setReferenceAttribution(event.target.value)} /></label>
                <p className="record-disclosure">Only mark uses that the source terms actually allow. Academic intent alone does not grant these rights. These choices record provenance; they do not authorize this project to train or upload data.</p>
                <div className="record-permissions">
                  {([
                    ['localDisplay', 'Local display'], ['training', 'Training reuse'], ['derivation', 'Create derivatives'],
                    ['traceViewing', 'Trace/reference viewing'], ['export', 'Export'], ['publicDemonstration', 'Public demonstration'], ['redistribution', 'Redistribution'],
                    ['cloudStorage', 'Cloud storage'], ['cloudProcessing', 'Cloud processing'],
                  ] as const).map(([key, label]) => <label key={key}><input type="checkbox" checked={referencePermittedUses[key]} onChange={(event) => setReferencePermittedUses((current) => ({ ...current, [key]: event.target.checked }))} /> Source terms permit: {label}</label>)}
                </div>
              </> : null}
            </fieldset>
            <fieldset className="record-fieldset"><legend>Creator and project consent</legend>
            <label><input type="checkbox" checked={creatorConfirmed} onChange={(event) => setCreatorConfirmed(event.target.checked)} /> I created this drawing and can grant the selected permissions.</label>
            <label><input type="checkbox" checked={canGrantPermissions} onChange={(event) => setCanGrantPermissions(event.target.checked)} /> I can grant the permissions I selected for the reference used.</label>
            <label><input type="checkbox" checked={referenceRightsDeclared} onChange={(event) => setReferenceRightsDeclared(event.target.checked)} /> I have checked the reference terms and declared its provenance and permitted uses above (or used no reference).</label>
            <label><input type="checkbox" checked={research} onChange={(event) => setResearch(event.target.checked)} /> I consent to this recording being stored locally for research and quality review.</label>
            <label><input type="checkbox" checked={evaluation} onChange={(event) => setEvaluation(event.target.checked)} /> I consent to evaluation.</label>
            <label><input type="checkbox" checked={training} onChange={(event) => setTraining(event.target.checked)} /> I separately consent to possible training reuse (optional; no training occurs in this phase).</label>
            <label><input type="checkbox" checked={publicDemo} onChange={(event) => setPublicDemo(event.target.checked)} /> I separately consent to public demonstration (optional).</label>
            <label><input type="checkbox" checked={cloudStorage} onChange={(event) => setCloudStorage(event.target.checked)} /> I separately consent to cloud storage (optional; nothing is uploaded by this recorder).</label>
            <label><input type="checkbox" checked={cloudProcessing} onChange={(event) => setCloudProcessing(event.target.checked)} /> I separately consent to cloud processing (optional; requires cloud-storage consent and is not enabled here).</label>
            </fieldset>
            <button className="button button--primary" onClick={() => { if (research && evaluation && creatorConfirmed && canGrantPermissions && referenceRightsDeclared && taskId.trim() && taskInstructions.trim() && contributorGroupId.trim() && (!cloudProcessing || cloudStorage) && (referenceKind === 'none' || !referencePermittedUses.cloudProcessing || referencePermittedUses.cloudStorage)) { setError(''); setPhase('ready') } else setError('Complete the task and pseudonymous contributor code, and confirm required creator, reference-rights, research, and evaluation fields. Cloud processing requires separate storage consent for both the contributor and referenced artwork.') }}>Continue</button>
          </> : null}
          {phase === 'ready' ? <><p>Review: {taskId} · {scope} · {style}. Drawings stay in this browser. Cloud and training choices are recorded as separate permissions and do not trigger transfer or training. Only an accepted sequence is stored.</p><button className="button button--primary" onClick={startRecording}>Begin recording</button><button className="button" onClick={() => setPhase('consent')}>Back</button></> : null}
          {phase === 'review' ? <><p>Elapsed {Math.round(elapsed / 1000)}s. Review the editable vector result before accepting it into the local research store.</p><button className="button button--primary" onClick={acceptRecording}>Accept and save locally</button><button className="button" onClick={exportBundle}>Export sequence JSON</button><button className="button" onClick={discardRecording}>Discard</button></> : null}
          {phase === 'accepted' ? <><p>The validated V3 recording is stored locally and can be exported for the evidence ledger. Local storage does not upload or authorize training.</p><button className="button" onClick={exportBundle}>Export sequence JSON</button><button className="button" onClick={discardRecording}>Record another</button></> : null}
          {error ? <p role="alert" className="operation-notice">{error}</p> : null}
        </div>
        {accepted.length ? <div className="research-panel"><h2>Local recordings ({accepted.length})</h2><p>V1/V2 records migrate for inspection, stay review-gated, and never inherit missing reference-level cloud permissions.</p><ul>{accepted.map((item) => <li key={item.recordingId}>{item.recordingId} · {item.statistics.strokeCount} strokes · {item.migrationReview ? 'legacy · needs review' : 'version 3'} <button className="button" onClick={async () => { await deleteAcceptedRecording(item.recordingId); setAccepted((current) => current.filter((candidate) => candidate.recordingId !== item.recordingId)) }}>Delete</button></li>)}</ul><button className="button" onClick={exportAcceptedBatch}>Export all local sequences</button></div> : null}
      </section>
    </main>
  )
}
