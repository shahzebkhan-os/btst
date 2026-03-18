"""
model_architecture.py
=====================
Neural Network model architectures for F&O prediction.

Primary Model:
  - Temporal Fusion Transformer (TFT) — purpose-built for multi-horizon financial time series

Ensemble Models:
  - TFT (primary)
  - LightGBM
  - XGBoost
  - Logistic Regression
  - Meta-learner (stacking)

All models use focal loss to handle class imbalance (UP/FLAT/DOWN).
"""

import logging
import warnings
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MultiLoss, SMAPE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")
logger = logging.getLogger("ModelArchitecture")

# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-class classification.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights [DOWN, FLAT, UP] (default: [0.25, 0.5, 0.25])
        gamma: Focusing parameter (default: 2.0)
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = config.FOCAL_ALPHA
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits from model
            targets: (N,) class indices (0=DOWN, 1=FLAT, 2=UP)

        Returns:
            Focal loss value
        """
        # Ensure alpha is on the same device as inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # Compute softmax probabilities
        p = torch.softmax(inputs, dim=1)

        # Get probabilities for the correct class
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL FUSION TRANSFORMER (TFT)
# ─────────────────────────────────────────────────────────────────────────────

class TFTModel:
    """
    Temporal Fusion Transformer model wrapper for F&O prediction.

    Key features:
      - Multi-head attention for interpretable feature importance
      - Variable selection networks for static and dynamic features
      - Gated residual networks for non-linear processing
      - Handles both static and time-varying features
    """

    def __init__(
        self,
        hidden_size: int = config.TFT_HIDDEN_SIZE,
        lstm_layers: int = config.TFT_LSTM_LAYERS,
        attention_heads: int = config.TFT_ATTENTION_HEADS,
        dropout: float = config.TFT_DROPOUT,
        learning_rate: float = config.TFT_LEARNING_RATE,
        max_encoder_length: int = config.TFT_MAX_ENCODER_LENGTH,
        max_prediction_length: int = config.TFT_MAX_PREDICTION_LENGTH,
    ):
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.model: Optional[TemporalFusionTransformer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None

    def create_dataset(
        self,
        df: pd.DataFrame,
        time_idx: str = "time_idx",
        target: str = "target",
        group_ids: List[str] = ["SYMBOL"],
        static_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        time_varying_known_categoricals: Optional[List[str]] = None,
        time_varying_known_reals: Optional[List[str]] = None,
        time_varying_unknown_categoricals: Optional[List[str]] = None,
        time_varying_unknown_reals: Optional[List[str]] = None,
    ) -> TimeSeriesDataSet:
        """
        Create TimeSeriesDataSet for TFT training.

        Args:
            df: Input DataFrame with all features
            time_idx: Time index column (sequential integer)
            target: Target variable column
            group_ids: Columns identifying time series groups
            static_categoricals: Static categorical features
            static_reals: Static real-valued features
            time_varying_known_categoricals: Future-known categorical features
            time_varying_known_reals: Future-known real features
            time_varying_unknown_categoricals: Unknown categorical features
            time_varying_unknown_reals: Unknown real features (most features)

        Returns:
            TimeSeriesDataSet object
        """
        # Set defaults if not provided
        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_known_categoricals is None:
            time_varying_known_categoricals = []
        if time_varying_known_reals is None:
            time_varying_known_reals = []
        if time_varying_unknown_categoricals is None:
            time_varying_unknown_categoricals = []
        if time_varying_unknown_reals is None:
            # Use all numeric columns except time_idx, target, and group_ids
            time_varying_unknown_reals = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in [time_idx, target] + group_ids
            ]

        dataset = TimeSeriesDataSet(
            df,
            time_idx=time_idx,
            target=target,
            group_ids=group_ids,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=group_ids, transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        return dataset

    def build_model(
        self,
        training_dataset: TimeSeriesDataSet,
        loss_fn: Optional[nn.Module] = None,
    ) -> TemporalFusionTransformer:
        """
        Build TFT model from dataset.

        Args:
            training_dataset: TimeSeriesDataSet for training
            loss_fn: Custom loss function (e.g., FocalLoss)

        Returns:
            TemporalFusionTransformer model
        """
        self.training_dataset = training_dataset

        if loss_fn is None:
            loss_fn = FocalLoss()

        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_head_size=self.attention_heads,
            dropout=self.dropout,
            hidden_continuous_size=config.TFT_HIDDEN_CONTINUOUS,
            output_size=config.NUM_CLASSES,  # 3 classes: DOWN, FLAT, UP
            loss=MultiLoss(metrics=[loss_fn]),
            log_interval=10,
            reduce_on_plateau_patience=config.TFT_REDUCE_ON_PLATEAU_PATIENCE,
            learning_rate=self.learning_rate,
        )

        self.model = model
        return model

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        max_epochs: int = config.MAX_EPOCHS,
        gpus: int = 0,
    ) -> Tuple[TemporalFusionTransformer, pl.Trainer]:
        """
        Train TFT model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            max_epochs: Maximum training epochs
            gpus: Number of GPUs (0 for CPU)

        Returns:
            Trained model and trainer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Configure callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=config.EARLY_STOP_PATIENCE,
            verbose=True,
            mode="min"
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if gpus > 0 else "cpu",
            devices=gpus if gpus > 0 else 1,
            gradient_clip_val=config.TFT_GRADIENT_CLIP_VAL,
            callbacks=[early_stop_callback, lr_monitor],
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        return self.model, trainer

    def predict(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        return_attention: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from TFT model.

        Args:
            test_dataloader: Test data loader
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optional attention weights
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(
            test_dataloader,
            mode="raw",
            return_x=True,
            return_y=True,
        )

        result = {
            "predictions": predictions.output.numpy(),
            "x": predictions.x,
            "y": predictions.y,
        }

        if return_attention and hasattr(predictions, "attention"):
            result["attention"] = predictions.attention.numpy()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Ensemble model combining TFT, LightGBM, XGBoost, and Logistic Regression
    with a meta-learner for stacking.

    Architecture:
      Level 0: TFT, LightGBM, XGBoost, Logistic Regression (base models)
      Level 1: Meta-learner (Logistic Regression / Ridge / Lasso)
    """

    def __init__(
        self,
        models: List[str] = config.ENSEMBLE_MODELS,
        meta_learner: str = config.META_LEARNER_TYPE,
    ):
        self.models = models
        self.meta_learner_type = meta_learner

        # Initialize base models
        self.tft_model: Optional[TFTModel] = None
        self.lgbm_model: Optional[lgb.LGBMClassifier] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.logreg_model: Optional[LogisticRegression] = None

        # Initialize meta-learner
        self.meta_learner: Optional[Any] = None
        self.scaler = StandardScaler()

    def build_base_models(self) -> None:
        """Build all base models."""
        if "tft" in self.models:
            self.tft_model = TFTModel()
            logger.info("TFT model initialized ✓")

        if "lgbm" in self.models:
            self.lgbm_model = lgb.LGBMClassifier(
                n_estimators=config.LGBM_N_ESTIMATORS,
                max_depth=config.LGBM_MAX_DEPTH,
                learning_rate=config.LGBM_LEARNING_RATE,
                num_leaves=config.LGBM_NUM_LEAVES,
                objective="multiclass",
                num_class=config.NUM_CLASSES,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            logger.info("LightGBM model initialized ✓")

        if "xgb" in self.models:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=config.XGB_LEARNING_RATE,
                subsample=config.XGB_SUBSAMPLE,
                objective="multi:softprob",
                num_class=config.NUM_CLASSES,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            logger.info("XGBoost model initialized ✓")

        if "logreg" in self.models:
            self.logreg_model = LogisticRegression(
                C=config.LOGREG_C,
                max_iter=config.LOGREG_MAX_ITER,
                multi_class="multinomial",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            logger.info("Logistic Regression model initialized ✓")

    def build_meta_learner(self) -> None:
        """Build meta-learner for stacking."""
        if self.meta_learner_type == "logreg":
            self.meta_learner = LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class="multinomial",
                random_state=42,
                n_jobs=-1,
            )
        elif self.meta_learner_type == "ridge":
            from sklearn.linear_model import RidgeClassifier
            self.meta_learner = RidgeClassifier(alpha=1.0, random_state=42)
        elif self.meta_learner_type == "lasso":
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

        logger.info(f"Meta-learner ({self.meta_learner_type}) initialized ✓")

    def train_base_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train all base models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Train LightGBM
        if self.lgbm_model is not None:
            logger.info("Training LightGBM...")
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.lgbm_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            logger.info("LightGBM training complete ✓")

        # Train XGBoost
        if self.xgb_model is not None:
            logger.info("Training XGBoost...")
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
            logger.info("XGBoost training complete ✓")

        # Train Logistic Regression
        if self.logreg_model is not None:
            logger.info("Training Logistic Regression...")
            # Scale features for LogReg
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.logreg_model.fit(X_train_scaled, y_train)
            logger.info("Logistic Regression training complete ✓")

    def get_base_predictions(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Get predictions from all base models.

        Args:
            X: Input features

        Returns:
            Stacked predictions from all base models (N, n_models * n_classes)
        """
        predictions = []

        if self.lgbm_model is not None:
            lgbm_pred = self.lgbm_model.predict_proba(X)
            predictions.append(lgbm_pred)

        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict_proba(X)
            predictions.append(xgb_pred)

        if self.logreg_model is not None:
            X_scaled = self.scaler.transform(X)
            logreg_pred = self.logreg_model.predict_proba(X_scaled)
            predictions.append(logreg_pred)

        # Stack predictions horizontally
        stacked_predictions = np.hstack(predictions)
        return stacked_predictions

    def train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train meta-learner on base model predictions.

        Args:
            X_train: Base model predictions (stacked)
            y_train: True labels
        """
        if self.meta_learner is None:
            self.build_meta_learner()

        logger.info("Training meta-learner...")
        base_preds_train = self.get_base_predictions(X_train)
        self.meta_learner.fit(base_preds_train, y_train)
        logger.info("Meta-learner training complete ✓")

    def predict(
        self,
        X: np.ndarray,
        return_probas: bool = True,
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Input features
            return_probas: Return probabilities (True) or class labels (False)

        Returns:
            Predictions from ensemble
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner() first.")

        base_preds = self.get_base_predictions(X)

        if return_probas:
            return self.meta_learner.predict_proba(base_preds)
        else:
            return self.meta_learner.predict(base_preds)


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test FocalLoss
    print("Testing Focal Loss...")
    focal_loss = FocalLoss(alpha=[0.25, 0.5, 0.25], gamma=2.0)
    inputs = torch.randn(10, 3, requires_grad=True)
    targets = torch.randint(0, 3, (10,))
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f} ✓")

    # Test TFT Model initialization
    print("\nTesting TFT Model...")
    tft_model = TFTModel()
    print("TFT Model initialized ✓")

    # Test Ensemble Model
    print("\nTesting Ensemble Model...")
    ensemble = EnsembleModel()
    ensemble.build_base_models()
    ensemble.build_meta_learner()
    print("Ensemble Model initialized ✓")

    print("\nAll model architecture tests passed! ✓")
