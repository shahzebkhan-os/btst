"""
explainability.py
=================
Explainability layer for the F&O ensemble predictor.
Provides SHAP values, TFT attention weights, and reasoning tags.
"""

import logging
import json
import datetime as dt
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import shap

import config

logger = logging.getLogger("ModelExplainer")

class ModelExplainer:
    """
    Handles model interpretability and reasoning generation.
    """

    def __init__(self):
        pass

    def compute_shap_values(self, lgbm_model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values using TreeExplainer for LightGBM.
        """
        logger.info("Computing SHAP values for LightGBM...")
        # Use TreeExplainer for fast and exact SHAP values
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X)
        
        # For multi-class, shap_values is a list of arrays (one per class)
        # We focus on the predicted class or a specific class (e.g., UP/DOWN)
        # Here we'll take the absolute mean across classes or just return the list
        
        # If it's a list, we'll convert it to a DataFrame of feature importance
        feature_names = X.columns.tolist()
        
        # Global importance: mean(|SHAP|)
        if isinstance(shap_values, list):
            # Sum absolute values across classes
            abs_shap = np.sum([np.abs(sv) for sv in shap_values], axis=0)
        else:
            abs_shap = np.abs(shap_values)
            
        global_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": np.mean(abs_shap, axis=0)
        }).sort_values("importance", ascending=False)
        
        logger.info(f"Top 10 features by SHAP:\n{global_importance.head(10)}")
        
        return pd.DataFrame(abs_shap, columns=feature_names)

    def get_top_shap_features(self, shap_values_row: np.ndarray, feature_names: List[str],
                          top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify top N features driving a single prediction.
        """
        # Get indices of top_n absolute SHAP values
        indices = np.argsort(np.abs(shap_values_row))[-top_n:][::-1]
        return [(feature_names[i], shap_values_row[i]) for i in indices]

    def extract_tft_attention(self, tft_wrapper: Any, dataloader: Any) -> pd.DataFrame:
        """
        Extract attention weights from TFT encoder.
        """
        logger.info("Extracting TFT attention weights...")
        if tft_wrapper.model is None:
            raise ValueError("TFT model not initialized")
            
        # Get raw interpretation from pytorch-forecasting
        interpretation = tft_wrapper.model.interpret_output(
            tft_wrapper.model.predict(dataloader, mode="raw"),
            reduction="mean"
        )
        
        attention = interpretation["attention"].cpu().numpy()
        # attention shape: (lookback_length,) or similar depending on reduction
        
        # Mocking lookback=20 context
        lookback = config.TFT_MAX_ENCODER_LENGTH
        attention_df = pd.DataFrame({
            "lookback_day": np.arange(lookback),
            "attention_weight": attention.flatten()[:lookback]
        })
        
        peak_day = attention_df.loc[attention_df["attention_weight"].idxmax(), "lookback_day"]
        logger.info(f"Peak attention on historical day: {peak_day}")
        
        return attention_df

    def generate_reasoning_tags(self, row: pd.Series, shap_top5: List[Tuple[str, float]]) -> str:
        """
        Generate human-readable reasoning tags based on feature triggers and SHAP importance.
        """
        tags = []
        
        # Mapping rules
        rules = {
            "pcr_flip_bullish": (lambda r: r.get("pcr_flip_bullish") == 1, "pcr_bullish_flip"),
            "pcr_extreme_bull": (lambda r: r.get("pcr_5d_zscore", 0) > 1.5, "high_pcr_fear"),
            "oi_buildup": (lambda r: r.get("oi_pct_chg", 0) > 2.0, "oi_long_buildup"),
            "spx_overnight_ret_pos": (lambda r: r.get("spx_ret", 0) > 0.01, "strong_usmkt_open"),
            "spx_overnight_ret_neg": (lambda r: r.get("spx_ret", 0) < -0.01, "usmkt_gap_down"),
            "vix_spike": (lambda r: r.get("vix_1d_chg", 0) > 5.0, "vix_spike_risk"),
            "dual_vix": (lambda r: r.get("vix_close", 0) > 25, "global_fear_mode"),
            "fii_buy": (lambda r: r.get("fii_net_cr", 0) > 1000, "fii_aggressive_buy"),
            "fii_sell": (lambda r: r.get("fii_net_cr", 0) < -1000, "fii_heavy_selling"),
            "rsi_oversold": (lambda r: r.get("rsi_14", 50) < 30, "rsi_oversold_bounce"),
            "rsi_overbought": (lambda r: r.get("rsi_14", 50) > 70, "rsi_overbought_fade"),
            "bb_squeeze": (lambda r: r.get("bb_width", 1) < 0.02, "vol_compression_breakout"),
        }
        
        for feat, (condition, tag) in rules.items():
            if condition(row):
                tags.append(tag)
                
        # Also include top SHAP features if they match certain prefixes
        for feat_name, shap_val in shap_top5:
            if "bulk" in feat_name and shap_val > 0: tags.append("smart_money_aligned")
            if "expiry" in feat_name: tags.append("expiry_max_pain_pin")
            
        # Deduplicate and limit to 6
        tags = list(dict.fromkeys(tags))[:6]
        return " | ".join(tags)

    def generate_signal_report(self, predictions_df: pd.DataFrame, feature_df: pd.DataFrame,
                           lgbm_model: Any) -> pd.DataFrame:
        """
        Enrich predictions with SHAP analysis and reasoning tags.
        """
        logger.info("Generating enriched signal report...")
        
        # 1. Compute SHAP
        feature_cols = [c for c in feature_df.columns if c not in ["DATE", "SYMBOL", "label_3c"]]
        shap_values = self.compute_shap_values(lgbm_model, feature_df[feature_cols])
        
        # 2. Add enriched fields
        report_data = []
        for i, (idx, pred_row) in enumerate(predictions_df.iterrows()):
            feat_row = feature_df.iloc[i]
            top5 = self.get_top_shap_features(shap_values.iloc[i].values, feature_cols)
            
            tags = self.generate_reasoning_tags(feat_row, top5)
            
            row_dict = pred_row.to_dict()
            row_dict.update({
                "reasoning_tags": tags,
                "top_feature_1": top5[0][0] if len(top5) > 0 else None,
                "top_feature_2": top5[1][0] if len(top5) > 1 else None,
                "risk_score": np.random.randint(1, 11) # Mock risk score
            })
            report_data.append(row_dict)
            
        report_df = pd.DataFrame(report_data)
        
        # Save output
        timestamp = dt.datetime.now().strftime("%Y%m%d")
        csv_path = config.OUTPUT_DIR / f"signals_{timestamp}.csv"
        json_path = config.OUTPUT_DIR / f"signals_{timestamp}.json"
        
        report_df.to_csv(csv_path, index=False)
        report_df.to_json(json_path, orient="records", indent=2)
        
        logger.info(f"Enriched report saved to {csv_path} and {json_path}")
        return report_df
