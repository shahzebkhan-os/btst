/**
 * Data Pipeline Status page
 */

import { useQuery } from '@tanstack/react-query'
import { getDataStatus } from '@/lib/api'
import { Database, CheckCircle2, AlertCircle, XCircle } from 'lucide-react'
import { timeAgo } from '@/lib/formatters'

export function DataPipeline() {
  const { data: status } = useQuery({
    queryKey: ['data', 'status'],
    queryFn: getDataStatus,
  })

  const onlineCount = status?.sources.filter(s => s.status === 'fresh').length || 0
  const totalCount = status?.sources.length || 0

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold mb-2">Data Pipeline</h2>
        <p className="text-textMuted">
          Sources Online: <span className="text-accentCyan">{onlineCount}</span> / {totalCount}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {status?.sources.map((source, idx) => (
          <div
            key={idx}
            className="bg-surface border border-surfaceElevated rounded-lg p-6"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <Database className="text-accentCyan" size={24} />
                <div>
                  <h3 className="font-semibold">{source.sourceName}</h3>
                  <p className="text-xs text-textMuted">
                    {source.lastUpdated ? timeAgo(source.lastUpdated) : 'Never updated'}
                  </p>
                </div>
              </div>

              {source.status === 'fresh' && <CheckCircle2 className="text-accentGreen" size={20} />}
              {source.status === 'stale' && <AlertCircle className="text-accentAmber" size={20} />}
              {source.status === 'error' && <XCircle className="text-accentRed" size={20} />}
            </div>

            <div className="flex gap-2">
              <span className="px-2 py-1 bg-surfaceElevated rounded text-xs">
                {source.rowCount.toLocaleString()} rows
              </span>
              {source.latencyMs && (
                <span className="px-2 py-1 bg-surfaceElevated rounded text-xs">
                  {source.latencyMs}ms
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
