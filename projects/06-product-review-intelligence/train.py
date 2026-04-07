"""Training entry point for the Product Review Intelligence project.

This script downloads the Amazon Electronics reviews dataset, runs a quick
sentiment-model smoke-test (to verify HuggingFace connectivity), and
optionally fine-tunes a lightweight classifier for star-rating prediction
from TF-IDF features.

Usage:
    python train.py [--sample-size N] [--no-finetune]

The core inference models (spaCy + cardiffnlp RoBERTa) are pre-trained and
do not require additional training.  This script validates the full pipeline
and optionally trains a local TF-IDF + LogisticRegression rating predictor
that can be used offline.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Setup / train Product Review Intelligence.")
    p.add_argument("--sample-size", type=int, default=50000,
                   help="Number of Amazon review records to download.")
    p.add_argument("--no-download", action="store_true",
                   help="Skip HuggingFace download (use already-ingested data).")
    p.add_argument("--no-finetune", action="store_true",
                   help="Skip TF-IDF rating-predictor training.")
    return p.parse_args()


def train_rating_classifier(csv_path: Path) -> None:
    """Train a lightweight TF-IDF + LogisticRegression star-rating predictor.

    Saves the model artefacts to artifacts/models/rating_classifier.pkl
    and artifacts/models/rating_vectorizer.pkl.

    Args:
        csv_path: Path to the ingested reviews CSV.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    logger.info("Training TF-IDF + LR rating predictor …")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "rating"])
    df["rating_int"] = df["rating"].astype(float).round().astype(int).clip(1, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str), df["rating_int"],
        test_size=0.2, random_state=42, stratify=df["rating_int"]
    )

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", class_weight="balanced")
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    logger.info("Rating predictor results:\n%s", classification_report(y_test, y_pred))

    out = Path("artifacts/models")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "rating_classifier.pkl")
    joblib.dump(vec, out / "rating_vectorizer.pkl")
    logger.info("Saved rating predictor artefacts to %s", out)


def verify_inference_pipeline() -> None:
    """Run a quick end-to-end smoke test on the prediction pipeline."""
    logger.info("Running inference pipeline smoke test …")
    from src.pipeline.prediction_pipeline import PredictionPipeline

    pipe = PredictionPipeline()
    test_review = (
        "The battery life on this phone is fantastic — lasts two full days. "
        "However the screen quality is disappointing for the price."
    )
    result = pipe.analyze(test_review)
    logger.info("Smoke test result: sentiment=%s rating=%d aspects=%d",
                result["overall_sentiment"], result["rating_prediction"],
                len(result["aspects"]))
    print("\nSmoke test passed.")
    print(f"  Overall sentiment : {result['overall_sentiment']}")
    print(f"  Rating prediction : {result['rating_prediction']}/5 stars")
    print(f"  Aspects found     : {len(result['aspects'])}")
    for asp in result["aspects"][:5]:
        print(f"    - {asp['aspect']}: {asp['sentiment']} ({asp['score']:.2f})")
    print(f"  Summary: {result['summary']}")


def main() -> None:
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)
    logger.info("=== Product Review Intelligence — Setup & Training ===")

    csv_path = Path("data/ingested/reviews_ingested.csv")

    # 1. Ingest (optional)
    if not args.no_download:
        logger.info("Step 1 — Data Ingestion")
        from src.components.data_ingestion import DataIngestion
        ingestion = DataIngestion(sample_size=args.sample_size)
        csv_path = ingestion.ingest()
    else:
        if not csv_path.exists():
            logger.error("Ingested data not found at %s. Run without --no-download.", csv_path)
            sys.exit(1)
        logger.info("Skipping download; using existing data at %s", csv_path)

    # 2. Train optional rating predictor
    if not args.no_finetune:
        logger.info("Step 2 — Training local TF-IDF rating predictor")
        train_rating_classifier(csv_path)
    else:
        logger.info("Skipping TF-IDF fine-tuning (--no-finetune).")

    # 3. Verify inference pipeline
    logger.info("Step 3 — Verifying inference pipeline")
    verify_inference_pipeline()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("Start the API    : uvicorn app:app --host 0.0.0.0 --port 8000")
    print("Start the UI     : streamlit run streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
