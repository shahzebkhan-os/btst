/**
 * Collapsible left sidebar navigation
 */

import { NavLink } from 'react-router-dom'
import { useState } from 'react'
import {
  BarChart2,
  Activity,
  Database,
  Brain,
  TrendingUp,
  AlertTriangle,
  Globe,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { StatusDot } from './StatusDot'
import { cn } from '@/lib/formatters'

const navItems = [
  { to: '/signals', icon: BarChart2, label: 'Signals' },
  { to: '/training', icon: Activity, label: 'Training' },
  { to: '/data-pipeline', icon: Database, label: 'Data Pipeline' },
  { to: '/explainability', icon: Brain, label: 'Explainability' },
  { to: '/backtester', icon: TrendingUp, label: 'Backtester' },
  { to: '/drift-health', icon: AlertTriangle, label: 'Drift Health' },
  { to: '/market-snapshot', icon: Globe, label: 'Market Snapshot' },
]

export function Sidebar() {
  const [isExpanded, setIsExpanded] = useState(true)

  return (
    <aside
      className={cn(
        'bg-surface border-r border-surfaceElevated transition-all duration-300 flex flex-col',
        isExpanded ? 'w-56' : 'w-16'
      )}
    >
      <div className="flex-1 py-6">
        <nav className="space-y-1 px-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                  'hover:bg-surfaceElevated',
                  isActive
                    ? 'text-accentCyan border-l-2 border-accentCyan bg-surfaceElevated'
                    : 'text-textMuted'
                )
              }
            >
              <item.icon size={20} />
              {isExpanded && <span className="text-sm font-medium">{item.label}</span>}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="p-4 border-t border-surfaceElevated space-y-3">
        <div className="flex items-center gap-2">
          <StatusDot status="active" size="sm" />
          {isExpanded && <span className="text-xs text-textMuted">Model v1.0.0</span>}
        </div>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full flex items-center justify-center p-2 rounded-lg hover:bg-surfaceElevated transition-colors text-textMuted"
        >
          {isExpanded ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>
      </div>
    </aside>
  )
}
