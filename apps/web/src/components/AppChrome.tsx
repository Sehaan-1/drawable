import type { PropsWithChildren } from 'react'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import { Link, useLocation } from 'react-router-dom'
import { ChevronDown, ClipboardCheck, FlaskConical, Gauge, HeartPulse, PenLine, Radio } from 'lucide-react'

const routes = [
  { to: '/draw', label: 'Draw', icon: PenLine },
  { to: '/curate', label: 'Curate', icon: FlaskConical },
  { to: '/benchmark', label: 'Benchmark', icon: Gauge },
  { to: '/setup', label: 'System', icon: HeartPulse },
  { to: '/record', label: 'Record', icon: Radio },
  { to: '/m0', label: 'M0', icon: ClipboardCheck },
]

export function WorkspaceSwitcher() {
  const location = useLocation()
  const current = routes.find((route) => location.pathname.startsWith(route.to)) ?? routes[0]!
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger className="workspace-switcher" aria-label="Switch workspace">
        <span className="wordmark">drawable</span>
        <span className="workspace-name">{current.label}</span>
        <ChevronDown size={14} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="start" sideOffset={6}>
          {routes.map((route) => {
            const Icon = route.icon
            return (
              <DropdownMenu.Item asChild key={route.to}>
                <Link className="menu-item" to={route.to}>
                  <Icon size={16} />
                  <span>{route.label}</span>
                </Link>
              </DropdownMenu.Item>
            )
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

export function ResearchShell({ title, eyebrow, badge = 'Fixture preview', children }: PropsWithChildren<{ title: string; eyebrow: string; badge?: string }>) {
  return (
    <div className="research-shell">
      <header className="research-topbar">
        <WorkspaceSwitcher />
        <div className="research-heading">
          <span>{eyebrow}</span>
          <strong>{title}</strong>
        </div>
        <span className="fixture-badge">{badge}</span>
      </header>
      <main className="research-main">{children}</main>
    </div>
  )
}
