"""Prediction pipeline: loads saved model and transformer for inference."""

import numpy as np
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.config.configuration import PredictionPipelineConfig
from src.utils.common import load_object, setup_logger

logger = setup_logger(__name__)


class PredictionPipeline:
    """Loads a persisted model and preprocessor and exposes methods for
    single-row and batch prediction."""

    def __init__(self, config: PredictionPipelineConfig | None = None) -> None:
        self.config = config or PredictionPipelineConfig()
        self._model = None
        self._transformer = None
        self._model_name: str = "unknown"

    def _load_artifacts(self) -> None:
        """Lazily load model and transformer from disk."""
        if self._model is None:
            logger.info("Loading model from %s", self.config.model_path)
            self._model = load_object(self.config.model_path)
        if self._transformer is None:
            logger.info("Loading transformer from %s", self.config.transformer_path)
            self._transformer = load_object(self.config.transformer_path)
        try:
            with open(self.config.model_name_path) as f:
                self._model_name = f.read().strip()
        except FileNotFoundError:
            self._model_name = type(self._model).__name__

    @property
    def model_name(self) -> str:
        self._load_artifacts()
        return self._model_name

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for a DataFrame of raw features.

        The DataFrame should contain the same columns as the original
        dataset (minus ``Weekly_Sales``).  Date features are extracted
        automatically.

        Args:
            input_data: DataFrame with raw feature columns.

        Returns:
            1-D numpy array of predicted Weekly_Sales values.
        """
        self._load_artifacts()

        df = input_data.copy()

        # Extract date features if Date column is present
        if "Date" in df.columns:
            df = DataTransformation.extract_date_features(df, "Date")

        # Convert IsHoliday to int
        if "IsHoliday" in df.columns:
            df["IsHoliday"] = df["IsHoliday"].astype(int)

        # Drop target column if accidentally included
        if "Weekly_Sales" in df.columns:
            df = df.drop(columns=["Weekly_Sales"])

        X = self._transformer.transform(df)
        predictions = self._model.predict(X)

        logger.info("Generated %d predictions.", len(predictions))
        return predictions

    def predict_single(self, features: dict) -> float:
        """Predict Weekly_Sales for a single record.

        Args:
            features: Dictionary mapping column names to values.

        Returns:
            Predicted Weekly_Sales as a float.
        """
        df = pd.DataFrame([features])
        preds = self.predict(df)
        return float(preds[0])
