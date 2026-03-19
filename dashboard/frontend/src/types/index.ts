/**
 * TypeScript type definitions matching FastAPI Pydantic models
 * These interfaces ensure type safety between frontend and backend
 */

/** Individual F&O signal prediction */
export interface Signal {
  symbol: string
  instrument: string
  expiryDt: string
  close: number
  direction: 'UP' | 'FLAT' | 'DOWN'
  confidence: number
  confidencePct: number
  predUpPct: number
  predDownPct: number
  expectedReturn?: number
  conformalLower?: number
  conformalUpper?: number
  openInt: number
  contracts: number
  dte: number
  liquidityPass: boolean
  positionSize?: number
  positionSizeLots?: number
  reasoning: string[]
}

/** Response for /api/signals/latest */
export interface SignalResponse {
  generatedAt: string
  modelVersion: string
  riskAppetiteScore: number
  signals: Signal[]
}

/** Historical signal performance */
export interface SignalHistory {
  date: string
  topSignalSymbol: string
  direction: 'UP' | 'FLAT' | 'DOWN'
  confidence: number
  wasCorrect?: boolean
  pnl?: number
}

/** Current training job status */
export interface TrainingStatus {
  isTraining: boolean
  currentEpoch?: number
  totalEpochs?: number
  currentFold?: number
  totalFolds?: number
  optunaTrial?: number
  optunaBestSharpe?: number
  phase?: 'optuna' | 'curriculum' | 'ensemble' | 'calibration'
  startedAt?: string
  estimatedCompletion?: string
  lastCheckpoint?: string
}

/** Training metrics for a single epoch */
export interface TrainingMetrics {
  epoch: number
  trainLoss: number
  valLoss: number
  trainAcc: number
  valAcc: number
  sharpe?: number
  fold: number
}

/** Status of a single data source */
export interface DataSourceStatus {
  sourceName: string
  lastUpdated: string
  rowCount: number
  status: 'fresh' | 'stale' | 'error'
  latencyMs?: number
}

/** Overall data pipeline status */
export interface DataStatus {
  sources: DataSourceStatus[]
}

/** Backtest metrics and results */
export interface BacktestResults {
  totalReturn: number
  cagr: number
  sharpe: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  totalTrades: number
  avgWin: number
  avgLoss: number
  monteCarlo?: Record<string, any>
}

/** Single point in equity curve */
export interface EquityPoint {
  date: string
  portfolioValue: number
  niftyValue: number
  drawdownPct: number
}

/** Global SHAP feature importance */
export interface ShapFeature {
  featureName: string
  importance: number
  direction: 'positive' | 'negative'
}

/** Feature contribution in SHAP waterfall */
export interface ShapFeatureContribution {
  name: string
  value: number
  shapValue: number
  contributionPct: number
}

/** Per-prediction SHAP waterfall data */
export interface ShapWaterfall {
  baseValue: number
  features: ShapFeatureContribution[]
}

/** Drifted feature information */
export interface DriftedFeature {
  featureName: string
  psi: number
  threshold: number
}

/** Model drift and health metrics */
export interface DriftHealth {
  modelStatus: 'HEALTHY' | 'DRIFT_DETECTED' | 'RETRAINING'
  daysSinceRetrain: number
  rolling7dAccuracy: number
  adwinStatus: 'Stable' | 'Drift Detected'
  topDriftedFeatures: DriftedFeature[]
}

/** Individual market data item */
export interface MarketItem {
  symbol: string
  name: string
  value: number
  change1d: number
  change5d: number
  category: string
}

/** Complete market snapshot grouped by category */
export interface MarketSnapshot {
  riskAppetiteScore: number
  usVix: number
  indiaVix: number
  dxy: number
  pcr: number
  markets: Record<string, MarketItem[]>
}

/** WebSocket connection status */
export type WSStatus = 'connected' | 'disconnected' | 'reconnecting'

/** API error response */
export interface ApiError {
  detail: string
  status: number
}
