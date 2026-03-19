/**
 * Status indicator dot with pulse animation
 */

import { cn } from '@/lib/formatters'

type StatusDotProps = {
  status: 'active' | 'idle' | 'error'
  size?: 'sm' | 'md' | 'lg'
  label?: string
}

export function StatusDot({ status, size = 'md', label }: StatusDotProps) {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4',
  }

  const colorClasses = {
    active: 'bg-accentGreen',
    idle: 'bg-accentAmber',
    error: 'bg-accentRed',
  }

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className={cn(sizeClasses[size], colorClasses[status], 'rounded-full')} />
        {status === 'active' && (
          <div className={cn(sizeClasses[size], colorClasses[status], 'rounded-full absolute inset-0 animate-pulse-glow')} />
        )}
      </div>
      {label && <span className="text-sm text-textMuted">{label}</span>}
    </div>
  )
}
