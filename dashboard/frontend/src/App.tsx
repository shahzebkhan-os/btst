/**
 * Main App component with routing
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { Signals } from './pages/Signals'
import { Training } from './pages/Training'
import { DataPipeline } from './pages/DataPipeline'
import { Explainability } from './pages/Explainability'
import { Backtester } from './pages/Backtester'
import { DriftHealth } from './pages/DriftHealth'
import { MarketSnapshot } from './pages/MarketSnapshot'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      refetchInterval: 60000, // 1 minute
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex h-screen bg-background">
          <Sidebar />
          <div className="flex flex-col flex-1">
            <TopBar />
            <main className="flex-1 overflow-auto p-6">
              <Routes>
                <Route path="/" element={<Navigate to="/signals" replace />} />
                <Route path="/signals" element={<Signals />} />
                <Route path="/training" element={<Training />} />
                <Route path="/data-pipeline" element={<DataPipeline />} />
                <Route path="/explainability" element={<Explainability />} />
                <Route path="/backtester" element={<Backtester />} />
                <Route path="/drift-health" element={<DriftHealth />} />
                <Route path="/market-snapshot" element={<MarketSnapshot />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
