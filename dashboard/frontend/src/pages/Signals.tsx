/**
 * Signals page - Today's top F&O signals and history
 */

import { useQuery } from '@tanstack/react-query'
import { getLatestSignals, getSignalsHistory } from '@/lib/api'
import { directionColor, directionArrow, formatConfidence, formatReturn, timeAgo } from '@/lib/formatters'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export function Signals() {
  const { data: signalsData, isLoading } = useQuery({
    queryKey: ['signals', 'latest'],
    queryFn: getLatestSignals,
  })

  const { data: history } = useQuery({
    queryKey: ['signals', 'history'],
    queryFn: getSignalsHistory,
  })

  if (isLoading) {
    return <div className="text-textMuted">Loading signals...</div>
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-display font-bold mb-4">Today's Signals</h2>
        <p className="text-textMuted mb-6">Generated {signalsData && timeAgo(signalsData.generatedAt)}</p>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {signalsData?.signals.map((signal, idx) => (
            <div
              key={idx}
              className="bg-surface border border-surfaceElevated rounded-lg p-6 hover:border-accentCyan transition-colors"
            >
              {/* Accent bar */}
              <div
                className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg ${
                  signal.direction === 'UP'
                    ? 'bg-accentCyan'
                    : signal.direction === 'DOWN'
                    ? 'bg-accentRed'
                    : 'bg-textMuted'
                }`}
              />

              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-display font-bold">{signal.symbol}</h3>
                  <p className="text-sm text-textMuted">{signal.expiryDt}</p>
                </div>
                <span className="px-2 py-1 bg-surfaceElevated rounded text-xs">
                  {signal.dte}D
                </span>
              </div>

              {/* Direction */}
              <div className="mb-4">
                <div className={`flex items-center gap-2 ${directionColor(signal.direction)}`}>
                  {signal.direction === 'UP' && <TrendingUp size={24} />}
                  {signal.direction === 'DOWN' && <TrendingDown size={24} />}
                  {signal.direction === 'FLAT' && <Minus size={24} />}
                  <span className="text-2xl font-bold">{signal.direction}</span>
                </div>
                <div className="mt-2 bg-surfaceElevated rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-accentCyan"
                    style={{ width: `${signal.confidence * 100}%` }}
                  />
                </div>
                <p className="text-sm text-textMuted mt-1">
                  {formatConfidence(signal.confidence)} confidence
                </p>
              </div>

              {/* Expected Return */}
              {signal.expectedReturn && (
                <div className="mb-4">
                  <span className="text-sm text-textMuted">Expected return: </span>
                  <span className="text-lg font-semibold">
                    {formatReturn(signal.expectedReturn)}
                  </span>
                  {signal.conformalLower && signal.conformalUpper && (
                    <p className="text-xs text-textMuted">
                      [{formatReturn(signal.conformalLower)} → {formatReturn(signal.conformalUpper)}]
                    </p>
                  )}
                </div>
              )}

              {/* Metrics */}
              <div className="flex gap-2 flex-wrap mb-4">
                <span className="px-2 py-1 bg-surfaceElevated rounded text-xs">
                  OI: {signal.openInt.toLocaleString()}
                </span>
                <span className="px-2 py-1 bg-surfaceElevated rounded text-xs">
                  Vol: {signal.contracts}
                </span>
              </div>

              {/* Reasoning tags */}
              {signal.reasoning.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {signal.reasoning.slice(0, 3).map((tag, i) => (
                    <span key={i} className="px-2 py-1 bg-surfaceElevated rounded text-xs font-mono text-accentCyan">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* History Table */}
      <div>
        <h2 className="text-xl font-display font-bold mb-4">Signal History</h2>
        <div className="bg-surface border border-surfaceElevated rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-surfaceElevated">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold">Date</th>
                <th className="px-4 py-3 text-left text-sm font-semibold">Symbol</th>
                <th className="px-4 py-3 text-left text-sm font-semibold">Direction</th>
                <th className="px-4 py-3 text-left text-sm font-semibold">Confidence</th>
                <th className="px-4 py-3 text-left text-sm font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {history?.slice(0, 10).map((item, idx) => (
                <tr key={idx} className="border-t border-surfaceElevated hover:bg-surfaceElevated/50">
                  <td className="px-4 py-3 text-sm">{item.date}</td>
                  <td className="px-4 py-3 text-sm font-mono">{item.topSignalSymbol}</td>
                  <td className={`px-4 py-3 text-sm ${directionColor(item.direction)}`}>
                    {directionArrow(item.direction)} {item.direction}
                  </td>
                  <td className="px-4 py-3 text-sm">{formatConfidence(item.confidence)}</td>
                  <td className="px-4 py-3 text-sm">
                    {item.wasCorrect === undefined ? (
                      <span className="text-textMuted">Pending</span>
                    ) : item.wasCorrect ? (
                      <span className="text-accentGreen">✓ Correct</span>
                    ) : (
                      <span className="text-accentRed">✗ Wrong</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
