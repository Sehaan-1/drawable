import type { ButtonHTMLAttributes, PropsWithChildren, ReactNode } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MissingProjectArtworkError } from '../lib/projectFiles'
import { useDocumentStore } from '../state/documentStore'
import { useUiStore } from '../state/uiStore'
import { WorkspaceDialogs } from './WorkspaceDialogs'

/**
 * The export dialog makes omissions impossible to miss: a PNG/SVG saved
 * without some artwork keeps the dialog open with a notice, and a project
 * that cannot round-trip without its missing artwork refuses — then offers
 * the omission as an explicit, user-chosen act.
 *
 * ``./primitives`` is stubbed: a real Radix dialog costs ~14s per test in
 * jsdom, and none of this behavior lives in Radix.
 */

const exports = vi.hoisted(() => ({ png: vi.fn(), svg: vi.fn(), project: vi.fn() }))

vi.mock('../lib/exportDocument', () => ({
  exportPng: (...args: unknown[]) => exports.png(...args),
  exportSvg: (...args: unknown[]) => exports.svg(...args),
}))

vi.mock('../lib/projectFiles', async () => {
  const actual = await vi.importActual<typeof import('../lib/projectFiles')>('../lib/projectFiles')
  return {
    ...actual,
    exportDrawableProject: (...args: unknown[]) => exports.project(...args),
    prepareImport: vi.fn(),
  }
})

vi.mock('../services/persistence', () => ({ stageImport: vi.fn() }))

vi.mock('./primitives', () => ({
  AppDialog: ({ open, title, children }: { open: boolean; title: string; children?: ReactNode }) =>
    open ? <div role="dialog" aria-label={title}>{children}</div> : null,
  Button: ({ children, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => <button type="button" {...props}>{children}</button>,
  Field: ({ children }: PropsWithChildren) => <div>{children}</div>,
}))

async function flush() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
  })
}

beforeEach(() => {
  exports.png.mockReset().mockResolvedValue([])
  exports.svg.mockReset().mockResolvedValue([])
  exports.project.mockReset().mockResolvedValue({ omitted: [] })
  useUiStore.setState({ settingsOpen: false, exportOpen: true, importOpen: false, shortcutsOpen: false })
  useDocumentStore.setState({ activeLayerId: 'layer-1' })
})

describe('export dialog omissions', () => {
  it('keeps the dialog open and says the PNG was saved without the missing artwork', async () => {
    exports.png.mockResolvedValue(['raster-gone'])
    render(<WorkspaceDialogs />)
    fireEvent.click(screen.getByRole('button', { name: /PNG · White background/ }))
    await screen.findByRole('status')
    expect(screen.getByRole('status').textContent).toContain('One imported artwork is no longer stored on this device')
    expect(screen.getByRole('status').textContent).toContain('saved without it')
    // An omission must not be invisible: closing the dialog now would hide it.
    expect(useUiStore.getState().exportOpen).toBe(true)
    expect(screen.getByRole('dialog', { name: 'Export drawing' })).toBeInTheDocument()
  })

  it('closes the dialog silently when an export kept everything', async () => {
    render(<WorkspaceDialogs />)
    fireEvent.click(screen.getByRole('button', { name: /^SVG/ }))
    await flush()
    expect(useUiStore.getState().exportOpen).toBe(false)
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('refuses a project that could not round-trip, names the ids, and offers explicit omission', async () => {
    exports.project.mockRejectedValue(new MissingProjectArtworkError(['raster-gone-abc']))
    render(<WorkspaceDialogs />)
    fireEvent.click(screen.getByRole('button', { name: /drawable project · Editable/ }))
    await screen.findByRole('alert')
    expect(screen.getByRole('alert').textContent).toContain('raster-gone-abc')
    expect(screen.getByText('Export project without missing artwork')).toBeInTheDocument()
    expect(useUiStore.getState().exportOpen).toBe(true)
  })

  it('lets the user choose the omission explicitly and then says what was left out', async () => {
    exports.project
      .mockRejectedValueOnce(new MissingProjectArtworkError(['raster-gone-abc']))
      .mockResolvedValueOnce({ omitted: ['raster-gone-abc'] })
    const document = useDocumentStore.getState().document
    render(<WorkspaceDialogs />)
    fireEvent.click(screen.getByRole('button', { name: /drawable project · Editable/ }))
    await screen.findByRole('alert')

    fireEvent.click(screen.getByText('Export project without missing artwork'))
    await screen.findByRole('status')

    expect(exports.project).toHaveBeenLastCalledWith(document, 'layer-1', { omitUnavailable: true })
    expect(screen.getByRole('status').textContent).toContain('saved without it')
    expect(screen.queryByRole('alert')).toBeNull()
    // Default attempt first, explicit opt-in second — never silent.
    expect(exports.project).toHaveBeenNthCalledWith(1, document, 'layer-1', { omitUnavailable: false })
  })
})
