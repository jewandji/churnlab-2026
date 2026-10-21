import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churnlab.pipeline import FEATURE_ORDER, build_pipeline, parse_inputs

# 1. Données et split
source = Path("data/raw/telco.csv")
sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
df = pd.read_csv(source)

train_df, remaining_df = train_test_split(
    df, test_size=0.4, stratify=df["Churn"], random_state=42
)
valid_df, test_df = train_test_split(
    remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=42
)

X_train = parse_inputs(train_df)[FEATURE_ORDER]
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

X_valid = parse_inputs(valid_df)[FEATURE_ORDER]
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

# 2. Entraînement du modèle retenu
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
pipeline = build_pipeline(model)
pipeline.fit(X_train, y_train)

# Métriques sur la validation
valid_probs = pipeline.predict_proba(X_valid)[:, 1]
ap = float(average_precision_score(y_valid, valid_probs))
roc_auc = float(roc_auc_score(y_valid, valid_probs))

# 3. Export de l'artefact complet
art_dir = Path("artifacts")
art_dir.mkdir(parents=True, exist_ok=True)
pipeline_path = art_dir / "churn_pipeline.joblib"
joblib.dump(pipeline, pipeline_path)
print(f"Artefact sauvegardé avec succès : {pipeline_path}")

# 4. Export des métadonnées de reproductibilité
metadata = {
    "source_data": {
        "file": source.as_posix(),
        "sha256": sha256,
        "split_seed": 42,
        "split_ratios": "60/20/20",
    },
    "features": {
        "ordered_inputs": FEATURE_ORDER,
        "target": "Churn (Yes=1, No=0)",
    },
    "selected_model": {
        "algorithm": "RandomForestClassifier",
        "parameters": {"n_estimators": 100, "max_depth": 8, "random_state": 42},
        "primary_metric": "average_precision",
        "validation_average_precision": round(ap, 4),
        "validation_roc_auc": round(roc_auc, 4),
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": version("scikit-learn"),
        "pandas": version("pandas"),
        "joblib": version("joblib"),
    },
}

meta_path = art_dir / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Métadonnées enregistrées : {meta_path}")