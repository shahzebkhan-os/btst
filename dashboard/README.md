# F&O Neural Network Dashboard

Real-time monitoring dashboard for the F&O Neural Network trading system. Visualizes live data loading, model training progress, signal output, SHAP explainability, backtester results, and model drift health.

## Architecture

```
React Frontend (Vite + TypeScript) ↔ WebSocket + REST ↔ FastAPI Backend ↔ Python ML Modules
         (localhost:5173)                              (localhost:8000)
```

## Features

### Backend (FastAPI)
- **REST API**: 12+ endpoints for signals, training status, data health, backtesting, SHAP, drift, and market data
- **WebSocket**: Real-time updates for training, data pipeline, signals, and drift channels
- **Auto-reconnect**: Exponential backoff WebSocket reconnection
- **CORS**: Enabled for local development

### Frontend (React 18 + TypeScript)
- **7 Pages**: Signals, Training Monitor, Data Pipeline, Explainability, Backtester, Drift Health, Market Snapshot
- **Real-time Updates**: WebSocket integration with React Query
- **Design System**: Terminal/trading aesthetic with JetBrains Mono + Syne fonts
- **Animations**: Framer Motion page transitions, pulse indicators, shimmer loading states
- **Charts**: Recharts for equity curves, loss curves, and metrics visualization

## Tech Stack

### Backend
- FastAPI 0.104+
- Uvicorn (ASGI server)
- Pydantic (data validation)
- WebSockets 12+
- Pandas (data processing)

### Frontend
- React 18.2
- TypeScript 5.2
- Vite 5.0 (build tool)
- TailwindCSS 3.3 (styling)
- React Router 6.20 (routing)
- TanStack Query 5.12 (data fetching + caching)
- Recharts 2.10 (charts)
- Framer Motion 10.16 (animations)
- Lucide React 0.294 (icons)

## Installation

### Prerequisites
- Python 3.11+
- Node.js 20+
- Running ML backend (predictor, trainer, drift monitor)

### Backend Setup

```bash
# Install Python dependencies
cd /path/to/btst
pip install -r requirements.txt

# Start FastAPI server
uvicorn dashboard.backend.api:app --reload --port 8000
```

The API will be available at:
- REST: http://localhost:8000/api
- WebSocket: ws://localhost:8000/ws/{channel}
- Docs: http://localhost:8000/docs

### Frontend Setup

```bash
# Install Node dependencies
cd dashboard/frontend
npm install

# Start development server
npm run dev
```

The dashboard will be available at: http://localhost:5173

### Quick Start (Both Services)

```bash
# From repository root
./dashboard/start.sh
```

This starts both FastAPI backend and Vite frontend in parallel.

## Configuration

### Backend Configuration

The backend reads data from these locations:
- `output/signals_*.json` - Daily signals
- `models/training_status.json` - Training status (create this file)
- `models/training_history.json` - Training metrics
- `output/backtest_results_*.json` - Backtest results
- `output/equity_curve_*.csv` - Equity curve data
- `output/shap_global_*.json` - SHAP feature importance
- `output/health_*.json` - Drift health metrics
- `data/extended/market_data_extended.parquet` - Market snapshot
- `data/` - All data sources (bhavcopy, VIX, FII/DII, option chains)

### Frontend Configuration

Environment variables (create `.env` file in `dashboard/frontend/`):

```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws
```

## Pages

### 1. Signals (`/signals`)
- Today's top-5 F&O signals with confidence, direction, expected return
- Signal history table (last 30 days)
- Accuracy tracking

### 2. Training Monitor (`/training`)
- Live training status (epoch, fold, phase, ETA)
- Loss curves (train vs val)
- Optuna trial progress
- Fold performance grid

### 3. Data Pipeline (`/data-pipeline`)
- 8 data source health cards
- Freshness indicators (fresh < 1h, stale 1-6h, error > 6h)
- Row counts and latency metrics
- Last 7 days timeline

### 4. Explainability (`/explainability`)
- Global SHAP feature importance (top-30)
- Horizontal bar chart with positive/negative direction
- Per-prediction SHAP waterfall (planned)

### 5. Backtester (`/backtester`)
- 6 key metrics: Total Return, CAGR, Sharpe, Max Drawdown, Win Rate, Profit Factor
- Equity curve chart (planned)
- Monthly returns heatmap (planned)
- Trade log table (planned)

### 6. Drift Health (`/drift-health`)
- Model status: HEALTHY | DRIFT_DETECTED | RETRAINING
- Days since retrain, rolling 7d accuracy
- ADWIN status
- Top drifted features with PSI scores

### 7. Market Snapshot (`/market-snapshot`)
- Risk appetite score gauge
- VIX, DXY, PCR indicators
- Market heatmap (100+ global signals grouped by category)
- Metals, Energy, Indices, Currencies, Bonds, Volatility, Crypto, Sectors

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn dashboard.backend.api:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/api/signals/latest
curl http://localhost:8000/api/training/status
curl http://localhost:8000/api/data/status
```

### Frontend Development

```bash
cd dashboard/frontend

# Development server with hot reload
npm run dev

# Type checking
npx tsc --noEmit

# Build for production
npm run build

# Preview production build
npm run preview
```

## WebSocket Channels

Connect to real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/training')
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  console.log('Training update:', data)
}
```

Available channels:
- `training` - Training status updates (every 2s during active training)
- `data_pipeline` - Data source health updates
- `signals` - New signal generation notifications
- `drift` - Model drift detection updates

## API Endpoints

### Signals
- `GET /api/signals/latest` - Today's top-5 signals
- `GET /api/signals/history` - Last 30 days of signals

### Training
- `GET /api/training/status` - Current training job status
- `GET /api/training/metrics` - Training metrics history

### Data
- `GET /api/data/status` - All data sources freshness status

### Backtest
- `GET /api/backtest/results` - Full backtest metrics
- `GET /api/backtest/equity_curve` - Equity curve data points

### Explainability
- `GET /api/shap/global` - Global feature importance (top-30)
- `GET /api/shap/prediction/{id}` - Per-prediction SHAP waterfall

### Drift
- `GET /api/drift/health` - Model drift and health metrics

### Market
- `GET /api/market/snapshot` - 100+ global market signals

## Design System

### Colors
- Background: `#07090D` (near-black)
- Surface: `#0D1117` (cards)
- Surface Elevated: `#131920`
- Accent Cyan: `#00E5CC` (primary)
- Accent Blue: `#4D9FFF`
- Accent Amber: `#F5A623`
- Accent Red: `#F05050`
- Accent Green: `#4ADE80`
- Text Primary: `#CDD5E0`
- Text Muted: `#58687A`
- Text Disabled: `#2E3D50`

### Typography
- **Data/Numbers**: JetBrains Mono (monospace)
- **Headings**: Syne (display font)

### Animations
- Fade-in on mount
- Number counter on value changes
- Pulse on live indicators
- Shimmer loading states

## Production Deployment

### Backend

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production settings
uvicorn dashboard.backend.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

```bash
cd dashboard/frontend

# Build for production
npm run build

# Serve with nginx, Apache, or any static file server
# Build output is in dashboard/frontend/dist/
```

### Environment Variables (Production)

Update frontend `.env.production`:
```env
VITE_API_URL=https://your-api-domain.com/api
VITE_WS_URL=wss://your-api-domain.com/ws
```

## Troubleshooting

### Backend Issues

**Q: "Module not found" errors**
A: Ensure you're in the repository root and `pip install -r requirements.txt` has been run.

**Q: "File not found" errors for data files**
A: The backend looks for files relative to the repository root. Ensure your ML pipeline is generating files in the expected locations.

### Frontend Issues

**Q: "Cannot connect to API"**
A: Check that the FastAPI backend is running on port 8000. Verify proxy settings in `vite.config.ts`.

**Q: WebSocket disconnects frequently**
A: Check firewall settings. WebSocket connections timeout after 30s of inactivity (ping messages are sent automatically).

**Q: TypeScript errors**
A: Run `npm install` to ensure all dependencies are installed. Check `tsconfig.json` settings.

## Known Limitations

1. **Mock Data**: Some endpoints return mock data if source files don't exist
2. **WebSocket Scaling**: Current implementation doesn't persist connections across server restarts
3. **Charts**: Some complex chart components are placeholders (planned for future enhancement)
4. **Authentication**: No authentication implemented (add JWT/OAuth for production)

## Future Features

1. Export functionality (CSV, JSON, markdown)
2. Keyboard shortcuts for navigation
3. Settings panel (API URL, refresh intervals, thresholds)
4. Toast notifications for important events
5. Advanced charts (Monte Carlo fan chart, attention heatmaps)
6. Mobile responsive design enhancements

## License

This dashboard is part of the F&O Neural Network Predictor project. Use at your own risk. Not financial advice.

## Support

For issues and questions:
1. Check logs in `logs/` directory
2. Review console output for errors
3. Verify data files exist and are valid JSON/CSV/Parquet format

---

**Built for production trading. Monitor with confidence.**
