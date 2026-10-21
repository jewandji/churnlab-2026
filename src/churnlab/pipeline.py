import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage sans état : convertit les espaces blancs de TotalCharges en NaN."""
    df_clean = data.copy()
    if "TotalCharges" in df_clean.columns:
        cleaned = df_clean["TotalCharges"].astype(str).str.strip()
        df_clean["TotalCharges"] = pd.to_numeric(
            cleaned.replace("", np.nan), errors="raise"
        )
    return df_clean


def build_pipeline(classifier=None) -> Pipeline:
    """Construit un pipeline vierge reliant prétraitement et modèle."""
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])