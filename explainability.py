"""
explainability.py
=================
Model explainability and interpretability for F&O predictions.

Methods:
  1. SHAP (SHapley Additive exPlanations) — feature importance and contribution
  2. Attention Weights Extraction (from TFT) — temporal feature importance
  3. Integrated Gradients (for TensorFlow models) — attribution analysis

SHAP provides:
  - Global feature importance (which features matter most overall)
  - Local explanations (why this specific prediction was made)
  - Feature interactions (how features work together)

Output Format:
  - Top-5 features contributing to each prediction
  - Reasoning tags (e.g., "High VIX", "Strong momentum", "OI buildup")
  - Confidence breakdown by feature group
"""

import logging
import warnings
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import config

warnings.filterwarnings("ignore")
logger = logging.getLogger("Explainability")

# Ensure output directory exists
config.OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP explainer for model predictions.

    SHAP assigns each feature an importance value for a particular prediction.
    The sum of SHAP values equals the difference between the model output
    and the expected output.

    Usage:
        explainer = SHAPExplainer(model)
        explainer.fit(X_background)
        shap_values = explainer.explain(X_test)
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "ensemble",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (sklearn-compatible)
            model_type: Type of model ('ensemble', 'tree', 'linear')
        """
        self.model = model
        self.model_type = model_type
        self.explainer: Optional[shap.Explainer] = None
        self.background_data: Optional[np.ndarray] = None

    def fit(
        self,
        X_background: np.ndarray,
        max_samples: int = config.SHAP_BACKGROUND_SIZE,
    ) -> "SHAPExplainer":
        """
        Fit SHAP explainer with background dataset.

        Args:
            X_background: Background dataset for SHAP
            max_samples: Maximum samples to use (for efficiency)

        Returns:
            Self
        """
        logger.info("Fitting SHAP explainer...")

        # Sample background data if too large
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            self.background_data = X_background[indices]
        else:
            self.background_data = X_background

        # Create explainer based on model type
        try:
            if self.model_type == "tree":
                # Use TreeExplainer for tree-based models (faster)
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for general models
                # Wrap predict_proba for classification
                if hasattr(self.model, "predict_proba"):
                    predict_fn = lambda x: self.model.predict_proba(x)
                else:
                    predict_fn = lambda x: self.model.predict(x)

                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    self.background_data,
                )

            logger.info("SHAP explainer fitted ✓")

        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise

        return self

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = config.SHAP_N_SAMPLES,
    ) -> np.ndarray:
        """
        Compute SHAP values for input samples.

        Args:
            X: Input features
            max_samples: Maximum samples to explain (for efficiency)

        Returns:
            SHAP values (n_samples, n_features, n_classes)
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit() first.")

        logger.info(f"Computing SHAP values for {len(X)} samples...")

        # Sample if too many samples
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(X_sample)
            logger.info("SHAP values computed ✓")
            return shap_values
        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            raise

    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 5,
        class_idx: int = 2,  # UP class
    ) -> List[Tuple[str, float]]:
        """
        Get top-K most important features for a single prediction.

        Args:
            shap_values: SHAP values for single sample (n_features, n_classes) or (n_features,)
            feature_names: List of feature names
            top_k: Number of top features to return
            class_idx: Class index for multi-class (default: UP=2)

        Returns:
            List of (feature_name, shap_value) tuples
        """
        # Handle multi-class SHAP values
        if len(shap_values.shape) == 2:
            shap_values_class = shap_values[:, class_idx]
        else:
            shap_values_class = shap_values

        # Get absolute SHAP values for importance ranking
        abs_shap = np.abs(shap_values_class)

        # Get top-K indices
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]

        # Create list of (feature, value) tuples
        top_features = [
            (feature_names[i], shap_values_class[i])
            for i in top_indices
        ]

        return top_features

    def generate_reasoning_tags(
        self,
        top_features: List[Tuple[str, float]],
    ) -> List[str]:
        """
        Generate human-readable reasoning tags from SHAP features.

        Args:
            top_features: List of (feature_name, shap_value) tuples

        Returns:
            List of reasoning tags
        """
        tags = []

        for feature_name, shap_value in top_features:
            # Determine direction
            direction = "High" if shap_value > 0 else "Low"

            # Create readable tag based on feature name
            if "vix" in feature_name.lower():
                tags.append(f"{direction} VIX")
            elif "rsi" in feature_name.lower():
                tags.append(f"{direction} RSI (momentum)")
            elif "macd" in feature_name.lower():
                tags.append(f"{'Bullish' if shap_value > 0 else 'Bearish'} MACD")
            elif "oi" in feature_name.lower() or "open_int" in feature_name.lower():
                tags.append(f"{'Strong' if shap_value > 0 else 'Weak'} OI buildup")
            elif "pcr" in feature_name.lower():
                tags.append(f"{'Bullish' if shap_value > 0 else 'Bearish'} PCR")
            elif "volume" in feature_name.lower():
                tags.append(f"{direction} volume")
            elif "atr" in feature_name.lower():
                tags.append(f"{direction} volatility")
            elif "ema" in feature_name.lower() or "sma" in feature_name.lower():
                tags.append(f"{'Uptrend' if shap_value > 0 else 'Downtrend'} (MA)")
            else:
                # Generic tag
                tags.append(f"{direction} {feature_name}")

        return tags

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        class_idx: int = 2,
        max_display: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot global feature importance.

        Args:
            shap_values: SHAP values for all samples
            X: Input features
            feature_names: List of feature names
            class_idx: Class index for multi-class
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
        """
        logger.info("Generating feature importance plot...")

        # Handle multi-class SHAP values
        if len(shap_values.shape) == 3:
            shap_values_class = shap_values[:, :, class_idx]
        else:
            shap_values_class = shap_values

        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_class,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved: {save_path}")

        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION WEIGHTS EXTRACTOR (TFT)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionWeightsExtractor:
    """
    Extract and visualize attention weights from Temporal Fusion Transformer.

    Attention weights show which time steps and features the model focuses on
    when making predictions.

    Usage:
        extractor = AttentionWeightsExtractor(tft_model)
        attention_weights = extractor.extract(test_data)
    """

    def __init__(
        self,
        tft_model: Any,
    ):
        self.tft_model = tft_model

    def extract(
        self,
        test_dataloader: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from TFT model.

        Args:
            test_dataloader: Test data loader

        Returns:
            Dictionary with attention weights
        """
        if not hasattr(self.tft_model, "predict"):
            raise ValueError("TFT model must have predict() method")

        logger.info("Extracting attention weights...")

        try:
            # Get predictions with attention
            predictions = self.tft_model.predict(
                test_dataloader,
                mode="raw",
                return_attention=True,
            )

            if hasattr(predictions, "attention"):
                attention_weights = predictions.attention.numpy()
                logger.info(f"Attention weights extracted: shape={attention_weights.shape} ✓")

                return {
                    "attention": attention_weights,
                    "decoder_attention": predictions.decoder_attention.numpy() if hasattr(predictions, "decoder_attention") else None,
                }
            else:
                logger.warning("No attention weights available in predictions")
                return {}

        except Exception as e:
            logger.error(f"Failed to extract attention weights: {e}")
            return {}

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot attention weights as heatmap.

        Args:
            attention_weights: Attention weights (n_samples, n_timesteps, n_features)
            feature_names: List of feature names (optional)
            sample_idx: Sample index to plot
            save_path: Path to save plot (optional)
        """
        logger.info("Generating attention heatmap...")

        if len(attention_weights.shape) < 3:
            logger.warning("Invalid attention weights shape")
            return

        # Get attention for single sample
        attention_sample = attention_weights[sample_idx]

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_sample.T,
            cmap="YlOrRd",
            xticklabels=range(attention_sample.shape[0]),
            yticklabels=feature_names if feature_names else range(attention_sample.shape[1]),
            cbar_kws={"label": "Attention Weight"},
        )
        plt.xlabel("Time Step")
        plt.ylabel("Feature")
        plt.title(f"TFT Attention Weights (Sample {sample_idx})")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Attention heatmap saved: {save_path}")

        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ExplainabilityManager:
    """
    Complete explainability pipeline for model predictions.

    Combines SHAP and attention weights to provide comprehensive explanations.

    Usage:
        manager = ExplainabilityManager(model, feature_names)
        manager.fit(X_background)
        explanations = manager.explain(X_test)
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "ensemble",
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type

        self.shap_explainer = SHAPExplainer(model, model_type)
        self.attention_extractor: Optional[AttentionWeightsExtractor] = None

    def fit(
        self,
        X_background: np.ndarray,
    ) -> "ExplainabilityManager":
        """
        Fit explainability methods.

        Args:
            X_background: Background dataset

        Returns:
            Self
        """
        logger.info("Fitting explainability methods...")

        # Fit SHAP
        self.shap_explainer.fit(X_background)

        logger.info("Explainability methods fitted ✓")
        return self

    def explain(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for predictions.

        Args:
            X: Input features
            sample_idx: Sample index to explain
            top_k: Number of top features

        Returns:
            Dictionary with explanations
        """
        logger.info(f"Generating explanation for sample {sample_idx}...")

        # Compute SHAP values
        shap_values = self.shap_explainer.explain(X)

        # Get single sample SHAP values
        if len(shap_values.shape) == 3:
            # Multi-class: (n_samples, n_features, n_classes)
            sample_shap = shap_values[sample_idx]
        else:
            # Binary/regression: (n_samples, n_features)
            sample_shap = shap_values[sample_idx]

        # Get top features
        top_features = self.shap_explainer.get_top_features(
            sample_shap,
            self.feature_names,
            top_k=top_k,
        )

        # Generate reasoning tags
        reasoning_tags = self.shap_explainer.generate_reasoning_tags(top_features)

        explanation = {
            "top_features": top_features,
            "reasoning_tags": reasoning_tags,
            "shap_values": sample_shap,
        }

        logger.info(f"Explanation generated ✓")
        logger.info(f"Reasoning tags: {', '.join(reasoning_tags)}")

        return explanation


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Test SHAP explainer
    print("Testing SHAP Explainer...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Create dummy data
    X_train, y_train = make_classification(n_samples=500, n_features=20, n_informative=10, n_classes=3, random_state=42)
    X_test, y_test = make_classification(n_samples=100, n_features=20, n_informative=10, n_classes=3, random_state=43)
    feature_names = [f"feature_{i}" for i in range(20)]

    # Train dummy model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Test SHAP
    shap_explainer = SHAPExplainer(model, model_type="tree")
    shap_explainer.fit(X_train, max_samples=100)
    shap_values = shap_explainer.explain(X_test, max_samples=10)
    print(f"SHAP values shape: {np.array(shap_values).shape} ✓")

    # Get top features for first sample
    if len(np.array(shap_values).shape) == 3:
        sample_shap = np.array(shap_values)[0]
    else:
        sample_shap = shap_values[0]

    top_features = shap_explainer.get_top_features(sample_shap, feature_names, top_k=5)
    print(f"Top features: {[f[0] for f in top_features]} ✓")

    # Generate reasoning tags
    tags = shap_explainer.generate_reasoning_tags(top_features)
    print(f"Reasoning tags: {tags} ✓")

    # Test Explainability Manager
    print("\nTesting Explainability Manager...")
    manager = ExplainabilityManager(model, feature_names, model_type="tree")
    manager.fit(X_train)
    explanation = manager.explain(X_test, sample_idx=0, top_k=5)
    print(f"Explanation: {len(explanation['reasoning_tags'])} tags ✓")

    print("\nAll explainability tests passed! ✓")
