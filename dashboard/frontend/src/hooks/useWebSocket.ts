/**
 * WebSocket hook for real-time updates
 */

import { useState, useEffect, useRef } from 'react'
import { WSClient } from '@/lib/websocket'
import type { WSStatus } from '@/types'

export function useWebSocket<T>(channel: string) {
  const [data, setData] = useState<T | null>(null)
  const [status, setStatus] = useState<WSStatus>('disconnected')
  const clientRef = useRef<WSClient | null>(null)

  useEffect(() => {
    const client = new WSClient(
      channel,
      (newData) => setData(newData),
      (newStatus) => setStatus(newStatus)
    )

    client.connect()
    clientRef.current = client

    return () => {
      client.disconnect()
    }
  }, [channel])

  return { data, status }
}
