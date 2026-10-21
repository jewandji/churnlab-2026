import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Chargement et parsing sans état
df = pd.read_csv("data/raw/telco.csv")

def parse_inputs(data: pd.DataFrame) -> pd.DataFrame:
    df_clean = data.copy()
    cleaned = df_clean["TotalCharges"].astype(str).str.strip()
    df_clean["TotalCharges"] = pd.to_numeric(cleaned.replace("", np.nan), errors="raise")
    return df_clean

# 2. Découpage stratifié 60/20/20 (seed=42)
train_df, remaining_df = train_test_split(
    df, test_size=0.4, stratify=df["Churn"], random_state=42
)
valid_df, test_df = train_test_split(
    remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=42
)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]
selected_columns = numeric_features + categorical_features

X_train = parse_inputs(train_df)[selected_columns]
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

X_valid = parse_inputs(valid_df)[selected_columns]
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

# 3. Fonction pour construire un pipeline frais
def build_pipeline(classifier) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

# 4. Modèles à évaluer
candidates = {
    "baseline_dummy": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
}

results = []

for name, clf in candidates.items():
    pipe = build_pipeline(clf)
    
    start_time = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_duration = time.perf_counter() - start_time
    
    preds = pipe.predict(X_valid)
    
    # Récupération des probabilités pour la classe positive (1)
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_valid)[:, 1]
        ap = average_precision_score(y_valid, probs)
        roc_auc = roc_auc_score(y_valid, probs)
    else:
        ap = np.nan
        roc_auc = np.nan
        
    metrics = {
        "model": name,
        "fit_time_seconds": round(fit_duration, 4),
        "average_precision": round(float(ap), 4),
        "roc_auc": round(float(roc_auc), 4),
        "accuracy": round(float(accuracy_score(y_valid, preds)), 4),
        "precision": round(float(precision_score(y_valid, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_valid, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_valid, preds, zero_division=0)), 4),
    }
    results.append(metrics)

# 5. Affichage et sauvegarde des résultats
res_df = pd.DataFrame(results).sort_values(by="average_precision", ascending=False)
print(res_df.to_string(index=False))

out_path = Path("evidence/tp02/model_comparison.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")