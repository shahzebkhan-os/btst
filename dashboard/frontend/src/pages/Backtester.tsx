/**
 * Backtester page
 */

import { useQuery } from '@tanstack/react-query'
import { getBacktestResults } from '@/lib/api'
import { formatReturn } from '@/lib/formatters'

export function Backtester() {
  const { data: results } = useQuery({
    queryKey: ['backtest', 'results'],
    queryFn: getBacktestResults,
  })

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold mb-2">Backtest Results</h2>
        <p className="text-textMuted">Historical performance metrics</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        {results && (
          <>
            <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
              <h3 className="text-sm text-textMuted mb-2">Total Return</h3>
              <p className="text-2xl font-bold text-accentCyan">
                {formatReturn(results.totalReturn / 100)}
              </p>
            </div>

            <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
              <h3 className="text-sm text-textMuted mb-2">CAGR</h3>
              <p className="text-2xl font-bold">{formatReturn(results.cagr / 100)}</p>
            </div>

            <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
              <h3 className="text-sm text-textMuted mb-2">Sharpe Ratio</h3>
              <p className="text-2xl font-bold">{results.sharpe.toFixed(2)}</p>
            </div>

            <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
              <h3 className="text-sm text-textMuted mb-2">Max Drawdown</h3>
              <p className="text-2xl font-bold text-accentRed">
                {formatReturn(results.maxDrawdown / 100)}
              </p>
            </div>

            <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
              <h3 className="text-sm text-textMuted mb-2">Win Rate</h3>
              <p className="text-2xl font-bold">{results.winRate.toFixed(1)}%</p>
            </div>

            <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
              <h3 className="text-sm text-textMuted mb-2">Profit Factor</h3>
              <p className="text-2xl font-bold">{results.profitFactor.toFixed(2)}</p>
            </div>
          </>
        )}
      </div>

      <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
        <h3 className="text-lg font-display font-bold mb-4">Equity Curve</h3>
        <p className="text-textMuted">Chart placeholder</p>
      </div>
    </div>
  )
}
