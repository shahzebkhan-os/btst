/**
 * Top navigation bar with system status indicators
 */

import { useQuery } from '@tanstack/react-query'
import { getLatestSignals, getMarketSnapshot } from '@/lib/api'
import { StatusDot } from './StatusDot'
import { timeAgo } from '@/lib/formatters'

export function TopBar() {
  const { data: signals } = useQuery({
    queryKey: ['signals', 'latest'],
    queryFn: getLatestSignals,
  })

  const { data: market } = useQuery({
    queryKey: ['market', 'snapshot'],
    queryFn: getMarketSnapshot,
  })

  return (
    <header className="bg-surface border-b border-surfaceElevated px-6 py-3">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-display font-bold text-accentCyan">F&O AI</h1>

        <div className="flex items-center gap-6">
          {/* Risk Appetite Score */}
          {market && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-textMuted">Risk Appetite</span>
              <div className="px-3 py-1 bg-surfaceElevated rounded-md">
                <span className="text-sm font-semibold text-accentCyan">
                  {market.riskAppetiteScore.toFixed(1)}/10
                </span>
              </div>
            </div>
          )}

          {/* India VIX */}
          {market && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-textMuted">India VIX</span>
              <div className="px-3 py-1 bg-surfaceElevated rounded-md">
                <span className="text-sm font-mono">{market.indiaVix.toFixed(1)}</span>
              </div>
            </div>
          )}

          {/* US VIX */}
          {market && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-textMuted">US VIX</span>
              <div className="px-3 py-1 bg-surfaceElevated rounded-md">
                <span className="text-sm font-mono">{market.usVix.toFixed(1)}</span>
              </div>
            </div>
          )}

          {/* Last Signal Time */}
          {signals && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-textMuted">Last signal:</span>
              <span className="text-sm">{timeAgo(signals.generatedAt)}</span>
            </div>
          )}

          {/* Live Status */}
          <StatusDot status="active" size="md" />
        </div>
      </div>
    </header>
  )
}
