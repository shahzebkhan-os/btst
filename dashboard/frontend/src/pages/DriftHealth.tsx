/**
 * Drift Health page
 */

import { useQuery } from '@tanstack/react-query'
import { getDriftHealth } from '@/lib/api'
import { AlertCircle, CheckCircle2 } from 'lucide-react'

export function DriftHealth() {
  const { data: health } = useQuery({
    queryKey: ['drift', 'health'],
    queryFn: getDriftHealth,
  })

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold mb-2">Model Health</h2>
        <p className="text-textMuted">Drift detection and performance monitoring</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Model Status</h3>
          <div className="flex items-center gap-2">
            {health?.modelStatus === 'HEALTHY' ? (
              <>
                <CheckCircle2 className="text-accentGreen" size={24} />
                <span className="text-xl font-bold text-accentGreen">HEALTHY</span>
              </>
            ) : (
              <>
                <AlertCircle className="text-accentRed" size={24} />
                <span className="text-xl font-bold text-accentRed">{health?.modelStatus}</span>
              </>
            )}
          </div>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Days Since Retrain</h3>
          <p className="text-3xl font-bold">{health?.daysSinceRetrain || 0}</p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">Rolling 7d Accuracy</h3>
          <p className="text-3xl font-bold">
            {((health?.rolling7dAccuracy || 0) * 100).toFixed(1)}%
          </p>
        </div>

        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-sm text-textMuted mb-2">ADWIN Status</h3>
          <p className={`text-xl font-bold ${
            health?.adwinStatus === 'Stable' ? 'text-accentGreen' : 'text-accentAmber'
          }`}>
            {health?.adwinStatus || 'Unknown'}
          </p>
        </div>
      </div>

      {health && health.topDriftedFeatures.length > 0 && (
        <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
          <h3 className="text-lg font-display font-bold mb-4">Top Drifted Features</h3>
          <div className="space-y-2">
            {health.topDriftedFeatures.map((feature, idx) => (
              <div key={idx} className="flex items-center justify-between">
                <span className="text-sm">{feature.featureName}</span>
                <span className="text-sm font-mono text-accentAmber">
                  PSI: {feature.psi.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
