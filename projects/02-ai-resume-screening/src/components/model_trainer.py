"""Model Trainer component: trains multiple classifiers, selects the best one."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.config.configuration import ModelTrainerConfig
from src.utils.common import get_logger, save_json, save_object

logger = get_logger(__name__)


class ModelTrainer:
    """Trains multiple classifiers and selects the best one by accuracy."""

    def __init__(self, config: ModelTrainerConfig | None = None):
        self.config = config or ModelTrainerConfig()

    def initiate_model_training(
        self, X_train, X_test, y_train, y_test
    ) -> tuple:
        """Train all candidate models, evaluate, and persist the best.

        Returns:
            Tuple of (best_model, best_model_name, model_scores).
        """
        logger.info("Starting model training")

        models = self._get_models()
        scores: dict[str, float] = {}

        for name, model in models.items():
            logger.info("Training %s ...", name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores[name] = round(acc, 4)
            logger.info("%s accuracy: %.4f", name, acc)

        # Select best
        best_name = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_model = models[best_name]

        logger.info(
            "Best model: %s with accuracy %.4f", best_name, scores[best_name]
        )

        # Save best model and report
        save_object(best_model, self.config.model_path)
        report = {
            "best_model": best_name,
            "best_accuracy": scores[best_name],
            "all_scores": scores,
        }
        save_json(report, self.config.model_report_path)

        return best_model, best_name, scores

    # ------------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------------

    def _get_models(self) -> dict:
        """Return a dictionary of candidate models."""
        rs = self.config.random_state

        # Wrap LinearSVC with CalibratedClassifierCV so we get predict_proba
        svc = LinearSVC(max_iter=2000, random_state=rs)
        calibrated_svc = CalibratedClassifierCV(svc, cv=3)

        return {
            "LinearSVC": calibrated_svc,
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=rs,
                n_jobs=-1,
            ),
            "MultinomialNB": MultinomialNB(alpha=0.1),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=rs,
                multi_class="multinomial",
                solver="lbfgs",
                n_jobs=-1,
            ),
        }
