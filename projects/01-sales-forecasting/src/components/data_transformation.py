"""Data transformation component: feature engineering, scaling, encoding."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config.configuration import DataTransformationConfig
from src.utils.common import create_directories, save_object, setup_logger

logger = setup_logger(__name__)


class DataTransformation:
    """Builds and applies a ``ColumnTransformer`` that handles numerical
    scaling, categorical encoding, missing-value imputation, and date-feature
    extraction for the Walmart sales dataset."""

    def __init__(self, config: DataTransformationConfig | None = None) -> None:
        self.config = config or DataTransformationConfig()

    # ------------------------------------------------------------------
    # Date feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Parse *date_col* and add Year, Month, WeekOfYear, DayOfWeek, Quarter.

        The original date column is dropped after extraction.
        """
        df = df.copy()
        dt = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
        df["Year"] = dt.dt.year
        df["Month"] = dt.dt.month
        df["WeekOfYear"] = dt.dt.isocalendar().week.astype(int)
        df["DayOfWeek"] = dt.dt.dayofweek
        df["Quarter"] = dt.dt.quarter
        df = df.drop(columns=[date_col])
        return df

    # ------------------------------------------------------------------
    # Transformer construction
    # ------------------------------------------------------------------

    def build_preprocessor(self) -> ColumnTransformer:
        """Create the ``ColumnTransformer`` with pipelines for numerical
        and categorical features.

        Numerical pipeline:
            1. Median imputation (handles MarkDown NaNs).
            2. Standard scaling.

        Categorical pipeline:
            1. Most-frequent imputation.
            2. Ordinal encoding (Type: A/B/C).

        Date-derived and boolean columns are numerical and included in the
        numerical pipeline after date extraction happens in ``transform_data``.
        """
        # After date extraction the numerical columns grow
        date_derived = ["Year", "Month", "WeekOfYear", "DayOfWeek", "Quarter"]
        all_numerical = self.config.numerical_columns + date_derived + [self.config.boolean_column]

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[["A", "B", "C"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, all_numerical),
                ("cat", categorical_pipeline, self.config.categorical_columns),
            ],
            remainder="drop",
        )

        return preprocessor

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Read train/test CSVs, engineer features, fit-transform, and persist
        the transformer.

        Args:
            train_path: Path to the training CSV.
            test_path: Path to the test CSV.

        Returns:
            ``(X_train, X_test, y_train, y_test, transformer_path)``
        """
        logger.info("Starting data transformation.")
        create_directories([self.config.root_dir])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info("Train shape before transformation: %s", train_df.shape)
        logger.info("Test shape before transformation: %s", test_df.shape)

        # --- Date feature extraction ---
        train_df = self.extract_date_features(train_df, self.config.date_column)
        test_df = self.extract_date_features(test_df, self.config.date_column)

        # --- Convert IsHoliday to int ---
        train_df[self.config.boolean_column] = train_df[self.config.boolean_column].astype(int)
        test_df[self.config.boolean_column] = test_df[self.config.boolean_column].astype(int)

        # --- Clip outliers in target ---
        target = self.config.target_column
        low = train_df[target].quantile(0.01)
        high = train_df[target].quantile(0.99)
        train_df[target] = train_df[target].clip(low, high)
        logger.info("Clipped Weekly_Sales to [%.2f, %.2f]", low, high)

        # --- Separate features and target ---
        y_train = train_df[target].values
        y_test = test_df[target].values
        X_train_df = train_df.drop(columns=[target])
        X_test_df = test_df.drop(columns=[target])

        # --- Build, fit, and transform ---
        preprocessor = self.build_preprocessor()
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)

        logger.info("X_train shape after transformation: %s", X_train.shape)
        logger.info("X_test shape after transformation: %s", X_test.shape)

        # --- Save transformer ---
        save_object(preprocessor, self.config.transformer_path)
        logger.info("Preprocessor saved to %s", self.config.transformer_path)

        logger.info("Data transformation completed.")
        return X_train, X_test, y_train, y_test, self.config.transformer_path
