"""
Model Trainer Component
Trains OneVsRestClassifier with LinearSVC and ClassifierChain with LogisticRegression.
Compares models by micro/macro F1 on the dev set and selects the best.
"""

import time
from typing import Tuple

import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.svm import LinearSVC

from src.config.configuration import ModelTrainerConfig, create_directories
from src.utils.common import get_logger, save_json, save_object

logger = get_logger(__name__)


class ModelTrainer:
    """Train and compare multi-label classifiers, select the best by micro/macro F1."""

    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        create_directories()

    def _train_ovr_linearsvc(
        self,
        X_train: sp.csr_matrix,
        y_train: sp.csr_matrix,
    ) -> OneVsRestClassifier:
        """Train a OneVsRestClassifier with LinearSVC."""
        logger.info(
            "Training OneVsRest + LinearSVC (C=%.3f, max_iter=%d)...",
            self.config.svc_C, self.config.svc_max_iter,
        )

        base_clf = LinearSVC(
            C=self.config.svc_C,
            max_iter=self.config.svc_max_iter,
            random_state=self.config.random_state,
            dual="auto",
        )
        ovr = OneVsRestClassifier(base_clf, n_jobs=-1)

        start = time.time()
        ovr.fit(X_train, y_train)
        elapsed = time.time() - start

        logger.info("OVR + LinearSVC trained in %.1f seconds.", elapsed)
        return ovr

    def _train_classifier_chain_lr(
        self,
        X_train: sp.csr_matrix,
        y_train,
    ) -> ClassifierChain:
        """Train a ClassifierChain with LogisticRegression."""
        logger.info(
            "Training ClassifierChain + LogisticRegression (C=%.3f, max_iter=%d)...",
            self.config.lr_C, self.config.lr_max_iter,
        )

        base_clf = LogisticRegression(
            C=self.config.lr_C,
            max_iter=self.config.lr_max_iter,
            solver=self.config.lr_solver,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        chain = ClassifierChain(
            base_clf,
            order="random",
            random_state=self.config.random_state,
        )

        # ClassifierChain requires dense y; convert if sparse
        if sp.issparse(y_train):
            y_train_dense = y_train.toarray()
        else:
            y_train_dense = y_train

        start = time.time()
        chain.fit(X_train, y_train_dense)
        elapsed = time.time() - start

        logger.info("ClassifierChain + LR trained in %.1f seconds.", elapsed)
        return chain

    def _evaluate_on_dev(
        self,
        model,
        X_dev: sp.csr_matrix,
        y_dev: sp.csr_matrix,
        model_name: str,
    ) -> dict:
        """Evaluate a model on the dev set and return micro/macro F1."""
        y_pred = model.predict(X_dev)

        if sp.issparse(y_dev):
            y_dev_dense = y_dev.toarray()
        else:
            y_dev_dense = y_dev

        if sp.issparse(y_pred):
            y_pred_dense = y_pred.toarray()
        else:
            y_pred_dense = y_pred

        micro_f1 = f1_score(y_dev_dense, y_pred_dense, average="micro", zero_division=0)
        macro_f1 = f1_score(y_dev_dense, y_pred_dense, average="macro", zero_division=0)

        logger.info(
            "%s -- Dev Micro-F1: %.4f, Macro-F1: %.4f",
            model_name, micro_f1, macro_f1,
        )

        return {
            "model_name": model_name,
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
        }

    def run(
        self,
        X_train: sp.csr_matrix,
        y_train: sp.csr_matrix,
        X_dev: sp.csr_matrix,
        y_dev: sp.csr_matrix,
    ) -> Tuple:
        """Train both models, evaluate, select best, and save."""
        logger.info("=== Model Training Started ===")

        # Train models
        ovr_model = self._train_ovr_linearsvc(X_train, y_train)
        chain_model = self._train_classifier_chain_lr(X_train, y_train)

        # Evaluate on dev
        ovr_metrics = self._evaluate_on_dev(ovr_model, X_dev, y_dev, "OVR_LinearSVC")
        chain_metrics = self._evaluate_on_dev(chain_model, X_dev, y_dev, "ClassifierChain_LR")

        # Save individual models
        save_object(ovr_model, self.config.ovr_model_path)
        save_object(chain_model, self.config.chain_model_path)
        logger.info("Individual models saved.")

        # Select best model by micro-F1 (primary metric)
        if ovr_metrics["micro_f1"] >= chain_metrics["micro_f1"]:
            best_model = ovr_model
            best_name = "OVR_LinearSVC"
            best_metrics = ovr_metrics
        else:
            best_model = chain_model
            best_name = "ClassifierChain_LR"
            best_metrics = chain_metrics

        logger.info("Best model: %s (Micro-F1=%.4f)", best_name, best_metrics["micro_f1"])

        save_object(best_model, self.config.best_model_path)
        logger.info("Best model saved to %s", self.config.best_model_path)

        # Save training report
        report = {
            "best_model": best_name,
            "models": {
                "OVR_LinearSVC": ovr_metrics,
                "ClassifierChain_LR": chain_metrics,
            },
            "selection_criterion": "micro_f1",
        }
        save_json(report, self.config.training_report_path)
        logger.info("Training report saved to %s", self.config.training_report_path)

        logger.info("=== Model Training Complete ===")
        return best_model, best_name, report
