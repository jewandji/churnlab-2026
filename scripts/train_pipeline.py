from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Chargement brut
df = pd.read_csv("data/raw/telco.csv")

# 2. Découpage stratifié 60/20/20 avec seed=42
train_df, remaining_df = train_test_split(
    df, test_size=0.4, stratify=df["Churn"], random_state=42
)
valid_df, test_df = train_test_split(
    remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=42
)

# 3. Fonction de parsing sans état pour TotalCharges (espace blanc -> NaN)
def parse_inputs(data: pd.DataFrame) -> pd.DataFrame:
    df_clean = data.copy()
    cleaned = df_clean["TotalCharges"].astype(str).str.strip()
    df_clean["TotalCharges"] = pd.to_numeric(cleaned.replace("", np.nan), errors="raise")
    return df_clean

# Préparation des jeux X et y
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]
selected_columns = numeric_features + categorical_features

X_train_raw = parse_inputs(train_df)[selected_columns]
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

X_valid_raw = parse_inputs(valid_df)[selected_columns]
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

# 4. Définition des branches du ColumnTransformer
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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Pipeline complet avec classifieur
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

# 6. Entraînement en un seul appel
pipe.fit(X_train_raw, y_train)

# 7. Prédictions directes sur les données de validation brutes
valid_preds = pipe.predict(X_valid_raw)

print("--- PIPELINE MODEL (6 features, Preprocessor + LogisticRegression) ---")
print("Accuracy:", accuracy_score(y_valid, valid_preds))
print("Confusion Matrix (labels [0, 1]):\n", confusion_matrix(y_valid, valid_preds, labels=[0, 1]))

# 8. Inspection de la statistique apprise (médianes)
imputer_stats = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].statistics_
print("\nStatistiques d'imputation apprises sur Train :")
for feat, stat in zip(numeric_features, imputer_stats):
    print(f"  - {feat} : {stat:.2f}")