/**
 * Training Monitor page
 */

import { useQuery } from '@tanstack/react-query'
import { getTrainingStatus, getTrainingMetrics } from '@/lib/api'

export function Training() {
  const { data: status } = useQuery({
    queryKey: ['training', 'status'],
    queryFn: getTrainingStatus,
  })

  const { data: metrics } = useQuery({
    queryKey: ['training', 'metrics'],
    queryFn: getTrainingMetrics,
  })

  if (!status?.isTraining) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <h2 className="text-2xl font-display font-bold mb-4">No Active Training</h2>
          <p className="text-textMuted">Last trained: {status?.lastCheckpoint || 'Never'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold mb-2">Training Monitor</h2>
        <p className="text-textMuted">Phase: {status.phase || 'Unknown'}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Epoch</h3>
          <p className="text-3xl font-bold">
            {status.currentEpoch} / {status.totalEpochs}
          </p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Fold</h3>
          <p className="text-3xl font-bold">
            {status.currentFold} / {status.totalFolds}
          </p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Best Sharpe</h3>
          <p className="text-3xl font-bold text-accentCyan">
            {status.optunaBestSharpe?.toFixed(2) || 'N/A'}
          </p>
        </div>
      </div>

      <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
        <h3 className="text-lg font-display font-bold mb-4">Training Metrics</h3>
        <p className="text-textMuted">Metrics: {metrics?.length || 0} epochs logged</p>
      </div>
    </div>
  )
}
