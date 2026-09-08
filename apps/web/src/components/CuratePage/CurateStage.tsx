/**
 * Image stage: shows the candidate's thumbnail, an optional ``line-art``
 * full-resolution swap, and the interactive crop overlay.
 *
 * The thumbnail URL is always preferred (it's smaller and ships in the
 * candidate payload). The line-art URL is what gets exported with the
 * snapshot, so we expose a toggle for reviewers who want to verify the
 * extraction quality against the original.
 *
 * A crop rectangle is no longer persisted with the label: committing a crop
 * cuts an **immutable child derivative** with its own files, processing, and
 * review state (``onCreateDerivative``). Skip advances the session queue
 * without a label.
 */

import { useEffect, useRef, useState } from 'react'
import {
  ChevronLeft,
  ChevronRight,
  Crop,
  Image as ImageIcon,
  PenLine,
  RotateCcw,
  Scissors,
  SkipForward,
} from 'lucide-react'
import { Button, IconButton } from '../primitives'
import { CropOverlay, type PixelRect, defaultCrop } from './CropOverlay'
import type { CurationCandidate } from './types'

export type StageView = 'thumbnail' | 'line-art'

export interface CurateStageProps {
  candidate: CurationCandidate | null
  editingCrop: boolean
  onToggleCrop: () => void
  crop: PixelRect | null
  onCropChange: (next: PixelRect) => void
  onCropCommit: (next: PixelRect) => void
  onPrev: () => void
  onNext: () => void
  onSkip?: () => void
  onReset: () => void
  /** Cut an immutable crop derivative from the current rectangle. */
  onCreateDerivative?: () => void
  /** A crop-derivative creation/processing request is in flight. */
  derivativePending?: boolean
  hasPrev: boolean
  hasNext: boolean
  position: { current: number; total: number } | null
  busy: boolean
}

export function CurateStage({
  candidate,
  editingCrop,
  onToggleCrop,
  crop,
  onCropChange,
  onCropCommit,
  onPrev,
  onNext,
  onSkip,
  onReset,
  onCreateDerivative,
  derivativePending = false,
  hasPrev,
  hasNext,
  position,
  busy,
}: CurateStageProps) {
  const [view, setView] = useState<StageView>('thumbnail')
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)

  // When the candidate changes, reset the view to the thumbnail and drop the
  // cached image size so the crop overlay does not animate to a stale rect.
  useEffect(() => {
    setView('thumbnail')
    setImageSize(null)
  }, [candidate?.asset_id])

  // Track the image's intrinsic size so the crop overlay can be drawn in
  // source-image pixel space. We use the rendered <img> element so any
  // browser-side scaling still maps to the right physical size.
  useEffect(() => {
    if (!candidate) return
    const img = imageRef.current
    if (!img) return
    const update = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        setImageSize({ width: img.naturalWidth, height: img.naturalHeight })
      }
    }
    if (img.complete) update()
    else img.addEventListener('load', update, { once: true })
    return () => img.removeEventListener('load', update)
  }, [candidate, view])

  // Default the crop to a centered 80% box the first time we see image
  // dimensions for a given candidate. We *don't* mutate the upstream
  // ``crop`` prop here — the parent owns the value and we only seed it.
  useEffect(() => {
    if (!candidate || !imageSize || crop) return
    onCropChange(defaultCrop(imageSize.width, imageSize.height))
  }, [candidate, imageSize, crop, onCropChange])

  const imageUrl = candidate ? (view === 'line-art' ? candidate.line_art_url : candidate.thumbnail_url) : null
  const alt = candidate?.asset_id ?? 'Curation candidate'

  return (
    <section className="candidate-stage" aria-label="Candidate preview">
      <div className="candidate-toolbar">
        <Button
          onClick={onToggleCrop}
          className={editingCrop ? 'is-active' : ''}
          disabled={!candidate}
          data-testid="toggle-crop"
        >
          <Crop size={15} /> {editingCrop ? 'Done crop' : 'Edit crop'}
        </Button>
        <Button onClick={onReset} disabled={!candidate || !editingCrop} data-testid="reset-crop">
          <RotateCcw size={15} /> Reset crop
        </Button>
        {onCreateDerivative ? (
          <Button
            onClick={onCreateDerivative}
            disabled={!candidate || !crop || derivativePending || busy}
            data-testid="create-derivative"
            title="Cut this rectangle into an immutable derivative with its own processing and review state"
          >
            <Scissors size={15} /> {derivativePending ? 'Creating…' : 'Create derivative'}
          </Button>
        ) : null}
        <div className="candidate-view-toggle" role="group" aria-label="Image source">
          <IconButton
            label="Show thumbnail"
            size="small"
            onClick={() => setView('thumbnail')}
            active={view === 'thumbnail'}
            disabled={!candidate}
          >
            <ImageIcon size={15} />
          </IconButton>
          <IconButton
            label="Show line art"
            size="small"
            onClick={() => setView('line-art')}
            active={view === 'line-art'}
            disabled={!candidate}
          >
            <PenLine size={15} />
          </IconButton>
        </div>
        <span className="candidate-toolbar-meta">
          {position ? `${position.current} of ${position.total}` : candidate ? '1 of 1' : 'No candidate'}
        </span>
      </div>

      <div className="candidate-image" data-testid="candidate-image">
        {imageUrl ? (
          <div className="candidate-frame">
            <img
              ref={imageRef}
              src={imageUrl}
              alt={alt}
              crossOrigin="anonymous"
              data-testid="candidate-img"
            />
            {imageSize ? (
              <CropOverlay
                crop={crop}
                imageWidth={imageSize.width}
                imageHeight={imageSize.height}
                editing={editingCrop}
                onChange={onCropChange}
                onCommit={onCropCommit}
              />
            ) : null}
          </div>
        ) : (
          <div className="candidate-empty">No candidate loaded</div>
        )}
      </div>

      <div className="candidate-nav">
        <Button onClick={onPrev} disabled={!hasPrev || busy} data-testid="prev-candidate">
          <ChevronLeft size={16} /> Previous
        </Button>
        <span>
          {candidate ? candidate.asset_id : '—'}
        </span>
        <div className="candidate-nav__group">
          {onSkip ? (
            <Button onClick={onSkip} disabled={!candidate || busy} data-testid="skip-candidate">
              <SkipForward size={15} /> Skip
            </Button>
          ) : null}
          <Button onClick={onNext} disabled={!hasNext || busy} data-testid="next-candidate">
            Next <ChevronRight size={16} />
          </Button>
        </div>
      </div>
    </section>
  )
}
