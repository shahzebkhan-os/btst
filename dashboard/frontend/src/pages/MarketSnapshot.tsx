/**
 * Market Snapshot page
 */

import { useQuery } from '@tanstack/react-query'
import { getMarketSnapshot } from '@/lib/api'
import { formatReturn } from '@/lib/formatters'

export function MarketSnapshot() {
  const { data: market } = useQuery({
    queryKey: ['market', 'snapshot'],
    queryFn: getMarketSnapshot,
  })

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold mb-2">Market Snapshot</h2>
        <p className="text-textMuted">Global market signals and risk indicators</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Risk Appetite</h3>
          <p className="text-3xl font-bold text-accentCyan">
            {market?.riskAppetiteScore.toFixed(1)}/10
          </p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">US VIX</h3>
          <p className="text-3xl font-bold">{market?.usVix.toFixed(1)}</p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">India VIX</h3>
          <p className="text-3xl font-bold">{market?.indiaVix.toFixed(1)}</p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">DXY</h3>
          <p className="text-3xl font-bold">{market?.dxy.toFixed(2)}</p>
        </div>
      </div>

      <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
        <h3 className="text-lg font-display font-bold mb-4">Market Heatmap</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
          {Object.entries(market?.markets || {}).map(([category, items]) =>
            items.slice(0, 6).map((item, idx) => (
              <div
                key={`${category}-${idx}`}
                className={`p-4 rounded-lg border ${
                  item.change1d > 0.02
                    ? 'bg-accentGreen/10 border-accentGreen'
                    : item.change1d < -0.02
                    ? 'bg-accentRed/10 border-accentRed'
                    : 'bg-surfaceElevated border-surfaceElevated'
                }`}
              >
                <p className="text-xs text-textMuted truncate">{item.name}</p>
                <p className="text-lg font-bold">{formatReturn(item.change1d)}</p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
