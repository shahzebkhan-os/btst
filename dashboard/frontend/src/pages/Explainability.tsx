/**
 * Explainability page - SHAP features
 */

import { useQuery } from '@tanstack/react-query'
import { getGlobalShap } from '@/lib/api'

export function Explainability() {
  const { data: shapFeatures } = useQuery({
    queryKey: ['shap', 'global'],
    queryFn: getGlobalShap,
  })

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold mb-2">Model Explainability</h2>
        <p className="text-textMuted">Global feature importance from SHAP analysis</p>
      </div>

      <div className="bg-surface border border-surfaceElevated rounded-lg p-6">
        <h3 className="text-lg font-display font-bold mb-4">Top Features</h3>
        <div className="space-y-2">
          {shapFeatures?.slice(0, 20).map((feature, idx) => (
            <div key={idx} className="flex items-center gap-4">
              <span className="text-sm text-textMuted w-48 truncate">{feature.featureName}</span>
              <div className="flex-1 bg-surfaceElevated rounded-full h-6 overflow-hidden">
                <div
                  className={`h-full ${
                    feature.direction === 'positive' ? 'bg-accentCyan' : 'bg-accentRed'
                  }`}
                  style={{ width: `${Math.min(feature.importance * 100, 100)}%` }}
                />
              </div>
              <span className="text-sm font-mono w-16 text-right">
                {feature.importance.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
