import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { M0EffortTracker } from './components/M0EffortTracker'

const DrawPage = lazy(() => import('./pages/DrawPage'))
const CuratePage = lazy(() => import('./pages/CuratePage'))
const BenchmarkPage = lazy(() => import('./pages/BenchmarkPage'))
const SetupPage = lazy(() => import('./pages/SetupPage'))
const RecordPage = lazy(() => import('./pages/RecordPage'))
const M0WorkspacePage = lazy(() => import('./pages/M0WorkspacePage'))

function AppLoading() {
  return <div className="app-loading" role="status">Opening workspace…</div>
}

export function App() {
  return (
    <Suspense fallback={<AppLoading />}>
      <Routes>
        <Route path="/" element={<Navigate to="/draw" replace />} />
        <Route path="/draw" element={<DrawPage />} />
        <Route path="/curate" element={<CuratePage />} />
        <Route path="/benchmark" element={<BenchmarkPage />} />
        <Route path="/setup" element={<SetupPage />} />
        <Route path="/record" element={<RecordPage />} />
        <Route path="/m0" element={<M0WorkspacePage />} />
        <Route path="*" element={<Navigate to="/draw" replace />} />
      </Routes>
      <M0EffortTracker />
    </Suspense>
  )
}
