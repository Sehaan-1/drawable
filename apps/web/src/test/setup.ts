import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// React Testing Library mounts components into a shared document; without
// an explicit cleanup between tests the DOM accumulates and queries like
// ``getByTestId`` start matching nodes from previous renders. ``afterEach``
// runs after every test in the file, even ones that don't use the library.
afterEach(() => {
  cleanup()
})

// jsdom ships with MouseEvent and TouchEvent but not PointerEvent. The
// CropOverlay uses pointer events for the draggable crop handles, and
// reading ``clientX`` from a missing PointerEvent produces ``NaN`` and
// silently breaks every drag in tests. Polyfill before importing the
// application code so the components see a complete event surface.
if (typeof PointerEvent === 'undefined') {
  class PointerEventPolyfill extends MouseEvent {
    public readonly pointerId: number
    public readonly pointerType: string
    public readonly isPrimary: boolean
    constructor(type: string, params: PointerEventInit = {}) {
      super(type, params)
      this.pointerId = params.pointerId ?? 1
      this.pointerType = params.pointerType ?? 'mouse'
      this.isPrimary = params.isPrimary ?? true
    }
  }
  // @ts-expect-error -- assign to the global PointerEvent so React's
  // synthetic event system can read clientX/clientY in tests.
  globalThis.PointerEvent = PointerEventPolyfill
}

// jsdom's Blob has no ``text()``/``arrayBuffer()``: its FileReader can read the
// bytes, so the polyfills go through it. Project export re-parses what it just
// wrote (checksums and all), which needs both.
if (typeof Blob !== 'undefined' && typeof Blob.prototype.arrayBuffer !== 'function') {
  Blob.prototype.arrayBuffer = function () {
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(this)
    })
  }
}
if (typeof Blob !== 'undefined' && typeof Blob.prototype.text !== 'function') {
  Blob.prototype.text = function () {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.onerror = () => reject(reader.error)
      reader.readAsText(this)
    })
  }
}

// jsdom's Crypto stops at getRandomValues/randomUUID; Node ships the standards
//-compliant SubtleCrypto the project round-trip needs for SHA-256. The module
// specifier is computed so the web project's types (no @types/node) stay clean.
if (typeof crypto !== 'undefined' && !crypto.subtle) {
  const nodeCrypto = 'node:' + 'crypto'
  const { webcrypto } = (await import(/* @vite-ignore */ nodeCrypto)) as { webcrypto: { subtle: SubtleCrypto } }
  Object.defineProperty(globalThis.crypto, 'subtle', { value: webcrypto.subtle })
}
