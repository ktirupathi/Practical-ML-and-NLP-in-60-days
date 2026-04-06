"""Prediction pipeline for document classification inference."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.components.data_ingestion import ID_TO_LABEL, LABEL_NAMES
from src.config.configuration import ConfigurationManager

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Loads a trained model and runs inference on document images.

    Accepts a PIL Image or file path, preprocesses it, and returns
    the predicted document class with confidence scores.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[ConfigurationManager] = None,
    ):
        self.config = config or ConfigurationManager()
        self.model_path = model_path or self.config.best_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image_processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and image processor from disk or HuggingFace."""
        model_source = self.model_path

        # If local path does not exist, fall back to HuggingFace model name
        if not Path(model_source).exists():
            logger.warning(
                "Local model not found at '%s'. Falling back to HuggingFace: '%s'",
                model_source,
                self.config.model_name,
            )
            model_source = self.config.model_name

        logger.info("Loading model from '%s'...", model_source)
        self.model = AutoModelForImageClassification.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()

        self.image_processor = AutoImageProcessor.from_pretrained(model_source)
        logger.info("Model loaded on device: %s", self.device)

    def predict(self, image: Union[str, Path, Image.Image]) -> Dict:
        """Classify a document image.

        Args:
            image: A file path (str/Path) or PIL Image object.

        Returns:
            Dictionary with 'document_type', 'confidence', and 'all_predictions'.
        """
        # Load image if path is given
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        probs = probabilities.squeeze().cpu().numpy()
        predicted_id = int(probs.argmax())
        confidence = float(probs[predicted_id])

        # Build full prediction mapping
        all_predictions = {
            LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
        }
        # Sort by confidence descending
        all_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )

        result = {
            "document_type": ID_TO_LABEL[predicted_id],
            "confidence": confidence,
            "all_predictions": all_predictions,
        }

        logger.info(
            "Prediction: %s (confidence: %.4f)",
            result["document_type"],
            result["confidence"],
        )
        return result
