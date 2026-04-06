"""Model trainer component: trains multiple regressors and selects the best."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.config.configuration import ModelTrainerConfig
from src.utils.common import create_directories, save_object, setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Trains XGBoost, LightGBM, and RandomForest regressors on the
    transformed training data, evaluates each on a validation set, and
    persists the best model by RMSE."""

    def __init__(self, config: ModelTrainerConfig | None = None) -> None:
        self.config = config or ModelTrainerConfig()

    def _build_models(self) -> dict:
        """Instantiate candidate models with configured hyperparameters."""
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor

        models = {
            "XGBRegressor": XGBRegressor(**self.config.xgb_params),
            "LGBMRegressor": LGBMRegressor(**self.config.lgbm_params),
            "RandomForestRegressor": RandomForestRegressor(**self.config.rf_params),
        }
        return models

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[object, str, dict[str, float]]:
        """Train all candidate models, compare RMSE, and save the best.

        Args:
            X_train: Transformed training features.
            X_test: Transformed test features.
            y_train: Training target values.
            y_test: Test target values.

        Returns:
            ``(best_model, best_model_name, results_dict)`` where
            *results_dict* maps model names to their test RMSE.
        """
        logger.info("Starting model training.")
        create_directories([self.config.root_dir])

        models = self._build_models()
        results: dict[str, float] = {}

        best_rmse = float("inf")
        best_model = None
        best_name = ""

        for name, model in models.items():
            logger.info("Training %s ...", name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            results[name] = rmse
            logger.info("%s -- RMSE: %.4f", name, rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name

        logger.info("Best model: %s (RMSE=%.4f)", best_name, best_rmse)

        # Save the best model
        save_object(best_model, self.config.model_path)
        logger.info("Best model saved to %s", self.config.model_path)

        # Save the model name for later reference
        with open(self.config.model_name_path, "w") as f:
            f.write(best_name)
        logger.info("Model name saved to %s", self.config.model_name_path)

        logger.info("Model training completed.")
        return best_model, best_name, results
