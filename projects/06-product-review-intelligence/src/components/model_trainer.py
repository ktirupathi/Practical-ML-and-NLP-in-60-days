"""Model trainer component: trains LightGBM, XGBoost, and Logistic Regression."""

from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config.configuration import AspectConfig, ModelTrainerConfig
from src.utils.common import load_artifact, save_artifact, setup_logger

logger = setup_logger("model_trainer")


class ModelTrainer:
    """Trains multiple classifiers and selects the best one by weighted F1."""

    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        self.best_model = None
        self.best_model_name = None
        self.all_scores = {}

    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM classifier."""
        logger.info("Training LightGBM...")
        model = lgb.LGBMClassifier(**self.config.lgbm_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="weighted")
        logger.info(f"LightGBM weighted F1: {score:.4f}")
        return model, score

    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier."""
        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(**self.config.xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="weighted")
        logger.info(f"XGBoost weighted F1: {score:.4f}")
        return model, score

    def _train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression classifier."""
        logger.info("Training Logistic Regression...")
        model = LogisticRegression(**self.config.lr_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="weighted")
        logger.info(f"Logistic Regression weighted F1: {score:.4f}")
        return model, score

    def initiate_model_training(
        self, train_path: Path, test_path: Path
    ) -> Tuple[Path, Dict]:
        """Train all models and save the best one.

        Args:
            train_path: Path to training data joblib file.
            test_path: Path to test data joblib file.

        Returns:
            Tuple of (best_model_path, scores_dict).
        """
        logger.info("Starting model training...")

        train_data = load_artifact(train_path)
        test_data = load_artifact(test_path)
        X_train, y_train = train_data["X"], train_data["y"]
        X_test, y_test = test_data["X"], test_data["y"]

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train all models
        models = {}

        lgbm_model, lgbm_score = self._train_lightgbm(X_train, y_train, X_test, y_test)
        models["lightgbm"] = (lgbm_model, lgbm_score)

        xgb_model, xgb_score = self._train_xgboost(X_train, y_train, X_test, y_test)
        models["xgboost"] = (xgb_model, xgb_score)

        lr_model, lr_score = self._train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        models["logistic_regression"] = (lr_model, lr_score)

        # Select best model
        self.all_scores = {name: score for name, (_, score) in models.items()}
        self.best_model_name = max(self.all_scores, key=self.all_scores.get)
        self.best_model = models[self.best_model_name][0]

        logger.info(f"Model scores: {self.all_scores}")
        logger.info(
            f"Best model: {self.best_model_name} "
            f"(F1={self.all_scores[self.best_model_name]:.4f})"
        )

        # Save all models
        models_dir = self.config.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, (model, _) in models.items():
            model_path = models_dir / f"{name}_model.joblib"
            save_artifact(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

        # Save best model reference
        best_model_path = models_dir / "best_model.joblib"
        save_artifact(self.best_model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

        # Save model scores
        from src.utils.common import save_json
        save_json(
            {"scores": self.all_scores, "best_model": self.best_model_name},
            models_dir / "model_scores.json",
        )

        return best_model_path, self.all_scores


class AspectExtractor:
    """Extracts product aspects from review text using keyword matching."""

    def __init__(self, config: AspectConfig = None):
        self.config = config or AspectConfig()

    def extract_aspects(self, text: str) -> Dict[str, str]:
        """Extract aspects and their sentiment from review text.

        Args:
            text: Cleaned review text.

        Returns:
            Dictionary mapping aspect names to sentiment ('positive', 'negative', 'neutral').
        """
        text_lower = text.lower()
        words = text_lower.split()
        detected_aspects = {}

        for aspect, keywords in self.config.aspect_keywords.items():
            # Check if any keyword for this aspect appears in the text
            found = False
            for keyword in keywords:
                if keyword in text_lower:
                    found = True
                    break

            if not found:
                continue

            # Determine sentiment for this aspect using context window
            aspect_sentiment = self._get_aspect_sentiment(text_lower, keywords)
            detected_aspects[aspect] = aspect_sentiment

        return detected_aspects

    def _get_aspect_sentiment(self, text: str, keywords: list) -> str:
        """Determine sentiment for a specific aspect using surrounding words.

        Args:
            text: Lowercased review text.
            keywords: List of keywords for the aspect.

        Returns:
            Sentiment string: 'positive', 'negative', or 'neutral'.
        """
        words = text.split()
        positive_score = 0
        negative_score = 0
        window_size = 5  # Words before and after aspect keyword

        for i, word in enumerate(words):
            is_keyword = any(kw in word for kw in keywords)
            if not is_keyword:
                continue

            # Check surrounding words
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context = words[start:end]

            for ctx_word in context:
                if ctx_word in self.config.positive_words:
                    positive_score += 1
                if ctx_word in self.config.negative_words:
                    negative_score += 1

        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"


if __name__ == "__main__":
    from src.config.configuration import DataTransformationConfig

    cfg = DataTransformationConfig()
    trainer = ModelTrainer()
    train_path = cfg.transformed_data_path / "train_data.joblib"
    test_path = cfg.transformed_data_path / "test_data.joblib"
    best_path, scores = trainer.initiate_model_training(train_path, test_path)
    print(f"Best model: {best_path}")
    print(f"Scores: {scores}")
