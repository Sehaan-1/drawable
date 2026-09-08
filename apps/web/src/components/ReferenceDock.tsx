import { useMemo, useRef } from 'react'
import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Image as ImageIcon,
  Layers3,
  Pin,
  PinOff,
  RefreshCw,
  Search,
  Sparkles,
  X,
} from 'lucide-react'
import { IconButton, StatusDot } from './primitives'
import { LayerInspector } from './LayerInspector'
import { useSearchStore } from '../state/searchStore'
import { useDocumentStore } from '../state/documentStore'
import { useUiStore } from '../state/uiStore'
import { useServiceStore } from '../services/serviceRegistry'
import type { InteractionEvent } from '@drawable/contracts'
import { UI_STYLE_TO_API } from '../services/liveServices'
import { traceSource } from '../lib/trace'
import type { ReferenceAsset, ReferenceGroup } from '../lib/types'

const styleOptions = ['Manga / anime', 'Western ink', 'Realistic', 'Cartoon', 'Gesture']

/**
 * Fire-and-forget interaction logging; failures never interrupt the artist.
 *
 * Each attempt carries its own `event_uuid`, so a retry (or a double-click
 * that produces the same interaction) is idempotent server-side and can never
 * inflate the learned profile. Style is *not* sent: the server derives it
 * from the gallery row.
 */
function recordInteraction(asset: ReferenceAsset, event: InteractionEvent) {
  const { services, sessionId } = useServiceStore.getState()
  const revision = useSearchStore.getState().response?.revision ?? 0
  void services.events
    .record({ event_uuid: crypto.randomUUID(), session_id: sessionId, asset_id: asset.id, event, query_revision: revision })
    .catch(() => undefined)
}

/**
 * Place a reference on the trace layer, if its source permissions allow it.
 *
 * Both entry points go through here, so the permission check cannot be
 * forgotten in one of them.
 */
function useTraceAction(asset: ReferenceAsset) {
  const setTrace = useDocumentStore((state) => state.setTrace)
  const source = traceSource(asset)
  return {
    allowed: source !== null,
    trace: () => {
      if (source === null) return
      setTrace(asset.id, source)
      recordInteraction(asset, 'trace')
    },
  }
}

function ServiceBadge() {
  const mode = useServiceStore((state) => state.mode)
  const health = useServiceStore((state) => state.health)
  if (mode === 'probing') return <span className="fixture-badge">Connecting…</span>
  if (mode === 'fixture') return <span className="fixture-badge" title={health?.message ?? ''}>Fixture</span>
  if (!health?.ready) return <span className="fixture-badge fixture-badge--error" title={health?.message ?? ''}>API not ready</span>
  if (health.mode === 'fixture') return <span className="fixture-badge fixture-badge--live" title={health.message}>API fixture</span>
  if (health.mode === 'cuda') return <span className="fixture-badge fixture-badge--live" title={health.message}>GPU</span>
  return <span className="fixture-badge fixture-badge--warn" title={health.message}>CPU fallback</span>
}

function ReferenceCard({ asset }: { asset: ReferenceAsset }) {
  const selected = useSearchStore((state) => state.selectedAsset?.id === asset.id)
  const pinned = useSearchStore((state) => state.pinned.some((item) => item.id === asset.id))
  const setSelectedAsset = useSearchStore((state) => state.setSelectedAsset)
  const togglePin = useSearchStore((state) => state.togglePin)
  const { allowed: canTrace, trace } = useTraceAction(asset)
  const open = () => { setSelectedAsset(asset); recordInteraction(asset, 'open') }
  // Pin state is durable and changes first; the learning event only follows a
  // state change that actually happened.
  const pin = () => { void togglePin(asset).then((isPinned) => recordInteraction(asset, isPinned ? 'pin' : 'unpin')) }

  return (
    <article className={`reference-card ${selected ? 'is-selected' : ''}`}>
      <button className="reference-card__media" onClick={open} aria-label={`View ${asset.title}`}>
        <img src={asset.imageUrl} alt={asset.title} draggable={false} />
        <span className="match-badge">{asset.match}</span>
      </button>
      <div className="reference-card__meta">
        <button className="reference-card__title" onClick={open}>{asset.title}</button>
        <span>{asset.style}</span>
      </div>
      <div className="reference-card__actions">
        <span>{asset.native ? 'Native' : 'Extracted'}</span>
        <IconButton label={pinned ? 'Unpin reference' : 'Pin reference'} size="small" onClick={pin}>
          {pinned ? <PinOff size={14} /> : <Pin size={14} />}
        </IconButton>
        <IconButton
          label={canTrace ? 'Place on trace layer' : 'Tracing is not permitted for this reference'}
          size="small"
          disabled={!canTrace}
          onClick={trace}
        >
          <Layers3 size={14} />
        </IconButton>
      </div>
    </article>
  )
}

function ReferenceSection({ group }: { group: ReferenceGroup }) {
  return (
    <section className="reference-section">
      <header>
        <div>
          <h3>{group.title}</h3>
          {group.tentative ? <span className="tentative-label">Tentative</span> : null}
        </div>
        <span>{group.results.length}</span>
      </header>
      <div className="reference-grid">
        {group.results.map((asset) => <ReferenceCard asset={asset} key={asset.id} />)}
      </div>
    </section>
  )
}

function ReferenceDetail({ asset }: { asset: ReferenceAsset }) {
  const setSelectedAsset = useSearchStore((state) => state.setSelectedAsset)
  const pinned = useSearchStore((state) => state.pinned.some((item) => item.id === asset.id))
  const togglePin = useSearchStore((state) => state.togglePin)
  const { allowed: canTrace, trace } = useTraceAction(asset)
  return (
    <section className="reference-detail">
      <header>
        <IconButton label="Back to reference grid" onClick={() => setSelectedAsset(null)}><ArrowLeft size={17} /></IconButton>
        <div><span className="dock-eyebrow">Selected reference</span><h2>{asset.title}</h2></div>
        <IconButton label="Close selected reference" onClick={() => setSelectedAsset(null)}><X size={17} /></IconButton>
      </header>
      <div className="reference-detail__media"><img src={asset.fullImageUrl ?? asset.imageUrl} alt={asset.title} /></div>
      <dl>
        <div><dt>Style</dt><dd>{asset.style}</dd></div>
        <div><dt>Scope</dt><dd>{asset.scope}</dd></div>
        <div><dt>Source</dt><dd>{asset.source}</dd></div>
        <div><dt>Artwork</dt><dd>{asset.native ? 'Native line art' : 'Extracted line art'}</dd></div>
        <div><dt>Tracing</dt><dd>{canTrace ? 'Permitted by the source' : 'Not permitted by the source'}</dd></div>
      </dl>
      <div className="reference-detail__actions">
        <button className="button" onClick={() => { void togglePin(asset).then((isPinned) => recordInteraction(asset, isPinned ? 'pin' : 'unpin')) }}>{pinned ? <PinOff size={15} /> : <Pin size={15} />}{pinned ? 'Unpin' : 'Pin'}</button>
        <button className="button button--primary" disabled={!canTrace} onClick={trace}><Layers3 size={15} />Trace</button>
        <a className="button" href={asset.fullImageUrl ?? asset.imageUrl} target="_blank" rel="noreferrer"><ExternalLink size={15} />Open</a>
      </div>
    </section>
  )
}

export function ReferenceDock() {
  const dockMode = useUiStore((state) => state.dockMode)
  const setDockMode = useUiStore((state) => state.setDockMode)
  const dockCollapsed = useUiStore((state) => state.dockCollapsed)
  const setDockCollapsed = useUiStore((state) => state.setDockCollapsed)
  const response = useSearchStore((state) => state.response)
  const loading = useSearchStore((state) => state.loading)
  const error = useSearchStore((state) => state.error)
  const textHint = useSearchStore((state) => state.textHint)
  const setTextHint = useSearchStore((state) => state.setTextHint)
  const selectedStyle = useSearchStore((state) => state.selectedStyle)
  const setSelectedStyle = useSearchStore((state) => state.setSelectedStyle)
  const selectedAsset = useSearchStore((state) => state.selectedAsset)
  const pinned = useSearchStore((state) => state.pinned)
  const revokedPins = useSearchStore((state) => state.revokedPins)
  const dismissRevokedPins = useSearchStore((state) => state.dismissRevokedPins)
  const pinError = useSearchStore((state) => state.pinError)
  const invalidate = useSearchStore((state) => state.invalidate)
  const scrollRef = useRef<HTMLDivElement>(null)

  const groups = useMemo(() => {
    if (!response) return []
    if (!selectedStyle || response.mode !== 'confident') return response.groups
    const best = response.groups.find((group) => group.id === 'best')
    const rest = response.groups.filter((group) => group.id !== 'best')
    const apiStyle = UI_STYLE_TO_API[selectedStyle]
    const matches = (group: ReferenceGroup) => group.title === selectedStyle || (apiStyle !== undefined && group.id === `style:${apiStyle}`)
    rest.sort((left, right) => Number(matches(right)) - Number(matches(left)))
    return best ? [best, ...rest] : rest
  }, [response, selectedStyle])

  if (dockMode === 'layers') return <aside className="reference-dock layer-dock"><LayerInspector /></aside>

  if (dockCollapsed) {
    return (
      <aside className="reference-dock reference-dock--collapsed">
        <button className="collapsed-dock-button" onClick={() => setDockCollapsed(false)}><ImageIcon size={18} /><span>References</span><ChevronRight size={16} /></button>
      </aside>
    )
  }

  const empty = !response || response.mode === 'empty'
  return (
    <aside className={`reference-dock ${selectedAsset ? 'has-selection' : ''}`}>
      <div className="reference-dock__toolbar">
        <div><span className="dock-eyebrow">Live copilot</span><h2>References</h2></div>
        <div className="reference-toolbar-actions">
          <IconButton label="Layers" onClick={() => setDockMode('layers')}><Layers3 size={17} /></IconButton>
          <IconButton label="Collapse references" onClick={() => setDockCollapsed(true)}><ChevronDown size={17} /></IconButton>
        </div>
      </div>
      <div className="reference-query">
        <label>
          <Search size={15} />
          <input value={textHint} onChange={(event) => setTextHint(event.target.value)} placeholder="Optional hint…" aria-label="Reference text hint" />
          {textHint ? <button aria-label="Clear hint" onClick={() => setTextHint('')}><X size={14} /></button> : null}
        </label>
        <div className="style-chips" aria-label="Preferred style">
          <button className={!selectedStyle ? 'is-active' : ''} onClick={() => setSelectedStyle(null)}>All</button>
          {styleOptions.map((style) => <button key={style} className={selectedStyle === style ? 'is-active' : ''} onClick={() => setSelectedStyle(style)}>{style.split(' ')[0]}</button>)}
        </div>
      </div>
      <div className="reference-status" role="status">
        <span><StatusDot tone={error ? 'error' : loading ? 'warning' : response?.mode === 'confident' ? 'success' : 'neutral'} />{error ? 'Search interrupted' : loading ? 'Looking at your drawing…' : response?.interpretation ?? 'Waiting for marks'}</span>
        <ServiceBadge />
      </div>
      {response?.warning && !error ? <p className="reference-warning" role="note">{response.warning}</p> : null}
      {revokedPins.length ? (
        <p className="reference-warning" role="note" data-testid="revoked-pins">
          {revokedPins.length === 1 ? '1 pinned reference is' : `${revokedPins.length} pinned references are`} no longer available and were removed.
          <button className="link-button" onClick={dismissRevokedPins}>Dismiss</button>
        </p>
      ) : null}
      {pinError ? <p className="reference-warning" role="note" data-testid="pin-error">{pinError}</p> : null}
      {response?.countsApproximate && !error && response.mode !== 'insufficient' ? (
        <p className="reference-warning" role="note" data-testid="approx-counts">
          {response.strokeStatus === 'absent'
            ? 'Searching a raster snapshot — no exact vector counts'
            : 'Vector counts are approximate'}
        </p>
      ) : null}
      {selectedAsset ? <ReferenceDetail asset={selectedAsset} /> : null}
      <div className="reference-scroll" ref={scrollRef}>
        {error ? (
          <div className="dock-empty"><RefreshCw size={22} /><h3>References are unavailable</h3><p>{error}</p><button className="button" onClick={() => invalidate()}>Try again</button></div>
        ) : empty ? (
          <div className="dock-empty"><Sparkles size={24} /><h3>Start drawing</h3><p>Reference ideas will appear here after each finished stroke.</p></div>
        ) : response?.mode === 'insufficient' ? (
          <div className="dock-empty"><ImageIcon size={24} /><h3>Keep drawing</h3><p>There isn’t enough information for a useful reference yet.</p></div>
        ) : (
          <>
            {groups.map((group) => <ReferenceSection group={group} key={group.id} />)}
            {pinned.length ? <ReferenceSection group={{ id: 'pinned', title: 'Pinned', results: pinned }} /> : null}
          </>
        )}
      </div>
    </aside>
  )
}

export function DockResizer() {
  const dockWidth = useUiStore((state) => state.dockWidth)
  const setDockWidth = useUiStore((state) => state.setDockWidth)
  const start = useRef<{ x: number; width: number } | null>(null)
  return (
    <div
      className="dock-resizer"
      role="separator"
      aria-label="Resize reference panel"
      aria-orientation="vertical"
      tabIndex={0}
      onPointerDown={(event) => {
        start.current = { x: event.clientX, width: dockWidth }
        event.currentTarget.setPointerCapture(event.pointerId)
      }}
      onPointerMove={(event) => {
        if (start.current) setDockWidth(start.current.width + start.current.x - event.clientX)
      }}
      onPointerUp={() => { start.current = null }}
      onKeyDown={(event) => {
        if (event.key === 'ArrowLeft') setDockWidth(dockWidth + 10)
        if (event.key === 'ArrowRight') setDockWidth(dockWidth - 10)
      }}
    />
  )
}
