"""
FastAPI backend for F&O Neural Network Trading Dashboard

This API bridges the Python ML backend to the React frontend, providing:
- REST endpoints for signals, training status, data health, backtesting, SHAP, drift, and market data
- WebSocket support for real-time updates (training, data_pipeline, signals, drift channels)
- CORS enabled for local development with Vite (localhost:5173)

Usage:
    uvicorn dashboard.backend.api:app --reload --port 8000
"""

import os
import json
import glob
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================================
# PYDANTIC MODELS (TypeScript-compatible camelCase field names)
# ============================================================================

class Signal(BaseModel):
    """Individual F&O signal prediction"""
    symbol: str = Field(..., description="Instrument symbol (e.g., NIFTY, BANKNIFTY)")
    instrument: str = Field(..., description="Full instrument name")
    expiryDt: str = Field(..., description="Expiry date")
    close: float = Field(..., description="Close price")
    direction: str = Field(..., description="Predicted direction: UP/FLAT/DOWN")
    confidence: float = Field(..., description="Model confidence (0-1)")
    confidencePct: float = Field(..., description="Confidence percentage")
    predUpPct: float = Field(..., description="UP probability %")
    predDownPct: float = Field(..., description="DOWN probability %")
    expectedReturn: Optional[float] = Field(None, description="Expected return %")
    conformalLower: Optional[float] = Field(None, description="Lower bound of conformal interval")
    conformalUpper: Optional[float] = Field(None, description="Upper bound of conformal interval")
    openInt: int = Field(..., description="Open interest")
    contracts: int = Field(..., description="Volume (contracts)")
    dte: int = Field(..., description="Days to expiry")
    liquidityPass: bool = Field(..., description="Liquidity filter pass/fail")
    positionSize: Optional[float] = Field(None, description="Recommended position size (₹)")
    positionSizeLots: Optional[int] = Field(None, description="Position size in lots")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning tags")


class SignalResponse(BaseModel):
    """Response for /api/signals/latest"""
    generatedAt: str = Field(..., description="ISO timestamp when signals were generated")
    modelVersion: str = Field(..., description="Model version identifier")
    riskAppetiteScore: float = Field(..., description="Risk appetite score (0-10)")
    signals: List[Signal] = Field(..., description="Top-5 signals")


class SignalHistory(BaseModel):
    """Historical signal performance"""
    date: str = Field(..., description="Signal date")
    topSignalSymbol: str = Field(..., description="Top signal symbol")
    direction: str = Field(..., description="Direction predicted")
    confidence: float = Field(..., description="Confidence")
    wasCorrect: Optional[bool] = Field(None, description="Was prediction correct")
    pnl: Optional[float] = Field(None, description="P&L if traded")


class TrainingStatus(BaseModel):
    """Current training job status"""
    isTraining: bool = Field(..., description="Whether training is active")
    currentEpoch: Optional[int] = Field(None, description="Current epoch number")
    totalEpochs: Optional[int] = Field(None, description="Total epochs planned")
    currentFold: Optional[int] = Field(None, description="Current CV fold")
    totalFolds: Optional[int] = Field(None, description="Total CV folds")
    optunanal: Optional[int] = Field(None, description="Current Optuna trial")
    optunaBestSharpe: Optional[float] = Field(None, description="Best Sharpe ratio found")
    phase: Optional[str] = Field(None, description="Training phase: optuna|curriculum|ensemble|calibration")
    startedAt: Optional[str] = Field(None, description="Training start time")
    estimatedCompletion: Optional[str] = Field(None, description="Estimated completion time")
    lastCheckpoint: Optional[str] = Field(None, description="Last checkpoint saved")


class TrainingMetrics(BaseModel):
    """Training metrics history"""
    epoch: int
    trainLoss: float
    valLoss: float
    trainAcc: float
    valAcc: float
    sharpe: Optional[float] = None
    fold: int


class DataSourceStatus(BaseModel):
    """Status of a single data source"""
    sourceName: str = Field(..., description="Data source name")
    lastUpdated: str = Field(..., description="Last update timestamp")
    rowCount: int = Field(..., description="Number of rows loaded")
    status: str = Field(..., description="Status: fresh|stale|error")
    latencyMs: Optional[float] = Field(None, description="Fetch latency in ms")


class DataStatus(BaseModel):
    """Overall data pipeline status"""
    sources: List[DataSourceStatus]


class BacktestResults(BaseModel):
    """Backtest metrics and results"""
    totalReturn: float = Field(..., description="Total return %")
    cagr: float = Field(..., description="Compound annual growth rate")
    sharpe: float = Field(..., description="Sharpe ratio")
    maxDrawdown: float = Field(..., description="Maximum drawdown %")
    winRate: float = Field(..., description="Win rate %")
    profitFactor: float = Field(..., description="Profit factor")
    totalTrades: int = Field(..., description="Total number of trades")
    avgWin: float = Field(..., description="Average winning trade %")
    avgLoss: float = Field(..., description="Average losing trade %")
    monteCarlo: Optional[Dict[str, Any]] = Field(None, description="Monte Carlo simulation results")


class EquityPoint(BaseModel):
    """Single point in equity curve"""
    date: str
    portfolioValue: float
    niftyValue: float
    drawdownPct: float


class ShapFeature(BaseModel):
    """Global SHAP feature importance"""
    featureName: str
    importance: float
    direction: str = Field(..., description="positive|negative")


class ShapWaterfall(BaseModel):
    """Per-prediction SHAP waterfall data"""
    baseValue: float
    features: List[Dict[str, Any]] = Field(..., description="List of {name, value, shapValue, contributionPct}")


class DriftHealth(BaseModel):
    """Model drift and health metrics"""
    modelStatus: str = Field(..., description="HEALTHY|DRIFT_DETECTED|RETRAINING")
    daysSinceRetrain: int
    rolling7dAccuracy: float
    adwinStatus: str = Field(..., description="Stable|Drift Detected")
    topDriftedFeatures: List[Dict[str, Any]] = Field(..., description="Features with high PSI scores")


class MarketItem(BaseModel):
    """Individual market data item"""
    symbol: str
    name: str
    value: float
    change1d: float
    change5d: float
    category: str


class MarketSnapshot(BaseModel):
    """Complete market snapshot grouped by category"""
    riskAppetiteScore: float
    usVix: float
    indiaVix: float
    dxy: float
    pcr: float
    markets: Dict[str, List[MarketItem]] = Field(..., description="Markets grouped by category")


# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections per channel"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "training": [],
            "data_pipeline": [],
            "signals": [],
            "drift": []
        }

    async def connect(self, websocket: WebSocket, channel: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove a WebSocket connection"""
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, data: dict):
        """Send JSON data to all subscribers of a channel"""
        if channel not in self.active_connections:
            return

        dead_connections = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(data)
            except Exception:
                dead_connections.append(connection)

        # Remove dead connections
        for connection in dead_connections:
            self.disconnect(connection, channel)


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="F&O Neural Network Dashboard API",
    description="Real-time monitoring API for F&O trading system",
    version="1.0.0"
)

# CORS middleware for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# Base paths for data
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_latest_file(pattern: str) -> Optional[Path]:
    """Get the most recent file matching a glob pattern"""
    files = list(BASE_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def read_json_file(file_path: Path) -> Optional[Dict]:
    """Read and parse a JSON file"""
    try:
        if file_path and file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "F&O Neural Network Dashboard API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Suppress favicon 404 logs"""
    from fastapi import Response
    return Response(status_code=204)


@app.get("/api/signals/latest", response_model=SignalResponse)
async def get_latest_signals():
    """Get today's top-5 F&O signals"""
    today = datetime.now().strftime("%Y%m%d")
    signal_file = get_latest_file(f"output/signals_{today}_*.json") or get_latest_file("output/signals_*.json")

    if not signal_file:
        # Return mock data if no signals file exists
        return SignalResponse(
            generatedAt=datetime.now().isoformat(),
            modelVersion="v1.0.0",
            riskAppetiteScore=6.5,
            signals=[]
        )

    data = read_json_file(signal_file)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to read signals file")

    # Parse signals - adjust field names from snake_case to camelCase
    signals = []
    for s in (data.get("signals", []) or data.get("data", []))[:5]:
        signals.append(Signal(
            symbol=s.get("SYMBOL", s.get("symbol", "")),
            instrument=s.get("INSTRUMENT", s.get("instrument", "")),
            expiryDt=s.get("EXPIRY_DT", s.get("expiry_dt", "")),
            close=float(s.get("CLOSE", s.get("close", 0))),
            direction=s.get("direction", "FLAT"),
            confidence=float(s.get("confidence", 0.5)),
            confidencePct=float(s.get("confidence_pct", 50.0)),
            predUpPct=float(s.get("pred_up_pct", 33.3)),
            predDownPct=float(s.get("pred_down_pct", 33.3)),
            expectedReturn=s.get("expected_return"),
            conformalLower=s.get("conformal_lower"),
            conformalUpper=s.get("conformal_upper"),
            openInt=int(s.get("OPEN_INT", s.get("open_int", 0))),
            contracts=int(s.get("CONTRACTS", s.get("contracts", 0))),
            dte=int(s.get("DTE", s.get("dte", 0))),
            liquidityPass=bool(s.get("liquidity_pass", True)),
            positionSize=s.get("position_size"),
            positionSizeLots=s.get("position_size_lots"),
            reasoning=s.get("reasoning", [])
        ))

    return SignalResponse(
        generatedAt=data.get("generated_at", datetime.now().isoformat()),
        modelVersion=data.get("model_version", "v1.0.0"),
        riskAppetiteScore=float(data.get("risk_appetite_score", 6.5)),
        signals=signals
    )


@app.get("/api/signals/history", response_model=List[SignalHistory])
async def get_signals_history():
    """Get last 30 days of signal accuracy"""
    history = []
    for days_ago in range(30):
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
        signal_files = list(BASE_DIR.glob(f"output/signals_{date}_*.json"))

        if signal_files:
            data = read_json_file(signal_files[0])
            if data and data.get("signals"):
                top_signal = data["signals"][0]
                history.append(SignalHistory(
                    date=date,
                    topSignalSymbol=top_signal.get("SYMBOL", top_signal.get("symbol", "")),
                    direction=top_signal.get("direction", "FLAT"),
                    confidence=float(top_signal.get("confidence", 0.5)),
                    wasCorrect=top_signal.get("was_correct"),
                    pnl=top_signal.get("pnl")
                ))

    return history


@app.get("/api/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training job status"""
    status_file = MODELS_DIR / "training_status.json"
    data = read_json_file(status_file)

    if not data:
        return TrainingStatus(
            isTraining=False,
            currentEpoch=None,
            totalEpochs=None,
            currentFold=None,
            totalFolds=None,
            optunaTrial=None,
            optunaBestSharpe=None,
            phase=None,
            startedAt=None,
            estimatedCompletion=None,
            lastCheckpoint=None
        )

    return TrainingStatus(
        isTraining=data.get("is_training", False),
        currentEpoch=data.get("current_epoch"),
        totalEpochs=data.get("total_epochs"),
        currentFold=data.get("current_fold"),
        totalFolds=data.get("total_folds"),
        optunaTrial=data.get("optuna_trial"),
        optunaBestSharpe=data.get("optuna_best_sharpe"),
        phase=data.get("phase"),
        startedAt=data.get("started_at"),
        estimatedCompletion=data.get("estimated_completion"),
        lastCheckpoint=data.get("last_checkpoint")
    )


@app.get("/api/training/metrics", response_model=List[TrainingMetrics])
async def get_training_metrics():
    """Get training metrics history"""
    metrics_file = MODELS_DIR / "training_history.json"
    data = read_json_file(metrics_file)

    if not data:
        return []

    metrics = []
    for m in data.get("metrics", []):
        metrics.append(TrainingMetrics(
            epoch=m["epoch"],
            trainLoss=m["train_loss"],
            valLoss=m["val_loss"],
            trainAcc=m["train_acc"],
            valAcc=m["val_acc"],
            sharpe=m.get("sharpe"),
            fold=m["fold"]
        ))

    return metrics


@app.get("/api/data/status", response_model=DataStatus)
async def get_data_status():
    """Get freshness status of all data sources"""
    sources = []

    # Define data sources to check
    data_sources = [
        {"name": "NSE Bhavcopy", "path": "data/bhavcopy/**/*.csv", "max_age_hours": 24},
        {"name": "India VIX", "path": "data/vix/india_vix.csv", "max_age_hours": 24},
        {"name": "FII/DII", "path": "data/fii_dii/fii_dii_data.csv", "max_age_hours": 24},
        {"name": "NIFTY Option Chain", "path": "data/option_chain/nifty_*.csv", "max_age_hours": 2},
        {"name": "BANKNIFTY Option Chain", "path": "data/option_chain/banknifty_*.csv", "max_age_hours": 2},
        {"name": "Global Markets", "path": "data/extended/market_data_extended.parquet", "max_age_hours": 6},
        {"name": "Intraday Candles", "path": "data/intraday/*.csv", "max_age_hours": 1},
        {"name": "Bulk/Block Deals", "path": "data/bulk_deals/*.csv", "max_age_hours": 24},
    ]

    for source in data_sources:
        latest_file = get_latest_file(source["path"])

        if latest_file:
            stat = latest_file.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            age_hours = (datetime.now() - last_modified).total_seconds() / 3600

            # Determine status
            if age_hours < source["max_age_hours"]:
                status = "fresh"
            elif age_hours < source["max_age_hours"] * 2:
                status = "stale"
            else:
                status = "error"

            # Count rows if CSV
            row_count = 0
            if latest_file.suffix == '.csv':
                try:
                    df = pd.read_csv(latest_file)
                    row_count = len(df)
                except:
                    pass
            elif latest_file.suffix == '.parquet':
                try:
                    df = pd.read_parquet(latest_file)
                    row_count = len(df)
                except:
                    pass

            sources.append(DataSourceStatus(
                sourceName=source["name"],
                lastUpdated=last_modified.isoformat(),
                rowCount=row_count,
                status=status,
                latencyMs=None
            ))
        else:
            sources.append(DataSourceStatus(
                sourceName=source["name"],
                lastUpdated="",
                rowCount=0,
                status="error",
                latencyMs=None
            ))

    return DataStatus(sources=sources)


@app.get("/api/backtest/results", response_model=BacktestResults)
async def get_backtest_results():
    """Get full backtest metrics"""
    results_file = get_latest_file("output/backtest_results_*.json")
    data = read_json_file(results_file)

    if not data:
        # Return mock data
        return BacktestResults(
            totalReturn=45.2,
            cagr=18.5,
            sharpe=1.84,
            maxDrawdown=-12.3,
            winRate=62.5,
            profitFactor=1.92,
            totalTrades=127,
            avgWin=3.2,
            avgLoss=-1.8,
            monteCarlo=None
        )

    return BacktestResults(
        totalReturn=data.get("total_return", 0),
        cagr=data.get("cagr", 0),
        sharpe=data.get("sharpe", 0),
        maxDrawdown=data.get("max_drawdown", 0),
        winRate=data.get("win_rate", 0),
        profitFactor=data.get("profit_factor", 0),
        totalTrades=data.get("total_trades", 0),
        avgWin=data.get("avg_win", 0),
        avgLoss=data.get("avg_loss", 0),
        monteCarlo=data.get("monte_carlo")
    )


@app.get("/api/backtest/equity_curve", response_model=List[EquityPoint])
async def get_equity_curve():
    """Get equity curve data"""
    curve_file = get_latest_file("output/equity_curve_*.csv")

    if not curve_file:
        return []

    try:
        df = pd.read_csv(curve_file)
        points = []
        for _, row in df.iterrows():
            points.append(EquityPoint(
                date=row.get("date", ""),
                portfolioValue=float(row.get("portfolio_value", 0)),
                niftyValue=float(row.get("nifty_value", 0)),
                drawdownPct=float(row.get("drawdown_pct", 0))
            ))
        return points
    except Exception as e:
        print(f"Error reading equity curve: {e}")
        return []


@app.get("/api/shap/global", response_model=List[ShapFeature])
async def get_global_shap():
    """Get global feature importance (top-30)"""
    shap_file = get_latest_file("output/shap_global_*.json")
    data = read_json_file(shap_file)

    if not data:
        return []

    features = []
    for f in data.get("features", [])[:30]:
        features.append(ShapFeature(
            featureName=f["feature_name"],
            importance=f["importance"],
            direction=f["direction"]
        ))

    return features


@app.get("/api/shap/prediction/{signal_id}", response_model=ShapWaterfall)
async def get_prediction_shap(signal_id: str):
    """Get per-prediction SHAP waterfall data"""
    # Mock data for now - would read from SHAP output files
    return ShapWaterfall(
        baseValue=0.51,
        features=[
            {"name": "vix_india", "value": 14.2, "shapValue": 0.18, "contributionPct": 25.0},
            {"name": "fii_net_buy", "value": 1250, "shapValue": 0.12, "contributionPct": 16.7},
            {"name": "rsi_14", "value": 68, "shapValue": -0.08, "contributionPct": -11.1},
        ]
    )


@app.get("/api/drift/health", response_model=DriftHealth)
async def get_drift_health():
    """Get model drift and health metrics"""
    today = datetime.now().strftime("%Y%m%d")
    health_file = get_latest_file(f"output/health_{today}.json") or get_latest_file("output/health_*.json")
    data = read_json_file(health_file)

    if not data:
        return DriftHealth(
            modelStatus="HEALTHY",
            daysSinceRetrain=3,
            rolling7dAccuracy=0.582,
            adwinStatus="Stable",
            topDriftedFeatures=[]
        )

    return DriftHealth(
        modelStatus=data.get("model_status", "HEALTHY"),
        daysSinceRetrain=data.get("days_since_retrain", 0),
        rolling7dAccuracy=data.get("rolling_7d_accuracy", 0.5),
        adwinStatus=data.get("adwin_status", "Stable"),
        topDriftedFeatures=data.get("top_drifted_features", [])
    )


@app.get("/api/market/snapshot", response_model=MarketSnapshot)
async def get_market_snapshot():
    """Get all 100+ global market signals grouped by category"""
    market_file = DATA_DIR / "extended" / "market_data_extended.parquet"

    if not market_file.exists():
        # Return mock data
        return MarketSnapshot(
            riskAppetiteScore=6.8,
            usVix=18.2,
            indiaVix=14.1,
            dxy=104.2,
            pcr=1.18,
            markets={}
        )

    try:
        df = pd.read_parquet(market_file)
        # Get latest row
        latest = df.iloc[-1].to_dict()

        # Group markets by category
        markets = {
            "metals": [],
            "energy": [],
            "indices": [],
            "currencies": [],
            "bonds": [],
            "volatility": [],
            "crypto": [],
            "sectors": []
        }

        # Parse market items (simplified - would need proper categorization)
        for col in df.columns:
            if col.startswith("price_"):
                symbol = col.replace("price_", "")
                category = "indices"  # Default category

                markets[category].append(MarketItem(
                    symbol=symbol,
                    name=symbol,
                    value=float(latest.get(col, 0)),
                    change1d=float(latest.get(f"change_1d_{symbol}", 0)),
                    change5d=float(latest.get(f"change_5d_{symbol}", 0)),
                    category=category
                ))

        return MarketSnapshot(
            riskAppetiteScore=float(latest.get("risk_appetite_score", 6.5)),
            usVix=float(latest.get("price_vix", 18.0)),
            indiaVix=float(latest.get("price_india_vix", 14.0)),
            dxy=float(latest.get("price_dxy", 104.0)),
            pcr=float(latest.get("nifty_pcr", 1.2)),
            markets=markets
        )
    except Exception as e:
        print(f"Error reading market snapshot: {e}")
        return MarketSnapshot(
            riskAppetiteScore=6.5,
            usVix=18.0,
            indiaVix=14.0,
            dxy=104.0,
            pcr=1.2,
            markets={}
        )


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """WebSocket endpoint for real-time updates"""
    if channel not in manager.active_connections:
        await websocket.close(code=1003)
        return

    await manager.connect(websocket, channel)

    try:
        # Send initial state immediately on connect
        if channel == "training":
            status = await get_training_status()
            await websocket.send_json(status.dict())
        elif channel == "data_pipeline":
            status = await get_data_status()
            await websocket.send_json(status.dict())
        elif channel == "signals":
            signals = await get_latest_signals()
            await websocket.send_json(signals.dict())
        elif channel == "drift":
            health = await get_drift_health()
            await websocket.send_json(health.dict())

        # Keep connection alive with ping every 30s
        while True:
            try:
                # Wait for client messages or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping
                await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, channel)


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    asyncio.create_task(broadcast_training_updates())


async def broadcast_training_updates():
    """Broadcast training updates every 2 seconds during active training"""
    while True:
        try:
            status_file = MODELS_DIR / "training_status.json"
            data = read_json_file(status_file)

            if data and data.get("is_training"):
                # Broadcast to training channel
                await manager.broadcast("training", data)
        except Exception as e:
            print(f"Error broadcasting training updates: {e}")

        await asyncio.sleep(2)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
