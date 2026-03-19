"""
model_tft.py
============
Temporal Fusion Transformer (TFT) implementation for F&O prediction.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import config
from .feature_engineering import FeatureEngineer

logger = logging.getLogger("TFTModel")

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-class classification.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else config.FOCAL_ALPHA
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute softmax probabilities
        p = torch.softmax(inputs, dim=1)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss

class TFTPredictor:
    """
    TFT model wrapper for F&O prediction.
    """

    def __init__(
        self,
        max_encoder_length: int = config.TFT_MAX_ENCODER_LENGTH,
        max_prediction_length: int = config.TFT_MAX_PREDICTION_LENGTH,
        hidden_size: int = config.TFT_HIDDEN_SIZE,
        lstm_layers: int = config.TFT_LSTM_LAYERS,
        attention_heads: int = config.TFT_ATTENTION_HEADS,
        learning_rate: float = config.TFT_LEARNING_RATE,
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.learning_rate = learning_rate
        
        self.model: Optional[TemporalFusionTransformer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None

    def prepare_dataset(self, df: pd.DataFrame, is_training: bool = True) -> TimeSeriesDataSet:
        if "time_idx" not in df.columns:
            df = df.sort_values(["SYMBOL", "DATE"]).copy()
            df["time_idx"] = df.groupby("SYMBOL").cumcount()

        eng = FeatureEngineer()
        # feature_cols = eng.get_feature_names(df)
        reals = [c for c in df.columns if c not in ["DATE", "SYMBOL", "label_3c", "time_idx"]]

        if is_training:
            dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="label_3c",
                group_ids=["SYMBOL"],
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=["SYMBOL"],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=reals,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True
            )
            self.training_dataset = dataset
        else:
            dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, df, predict=True)
        return dataset

    def build_model(self, training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_head_size=self.attention_heads,
            output_size=3, 
            loss=MultiLoss(metrics=[FocalLoss()]),
            learning_rate=self.learning_rate,
        )
        return self.model

    def train(self, train_dl, val_dl, max_epochs: int = config.MAX_EPOCHS):
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
        )
        trainer.fit(self.model, train_dl, val_dl)
        return trainer

    def predict(self, dataloader) -> np.ndarray:
        self.model.eval()
        raw = self.model.predict(dataloader, mode="raw")
        probas = torch.softmax(raw.output, dim=-1).squeeze(1).detach().cpu().numpy()
        return probas
