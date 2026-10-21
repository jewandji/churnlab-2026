from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Chargement des données
source = Path("data/raw/telco.csv")
df = pd.read_csv(source)

# 2. Découpage stratifié 60/20/20
train_df, remaining_df = train_test_split(
    df, test_size=0.4, stratify=df["Churn"], random_state=42
)
valid_df, test_df = train_test_split(
    remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=42
)

# 3. Vérification de non-chevauchement des identifiants
train_ids = set(train_df["customerID"])
valid_ids = set(valid_df["customerID"])
test_ids = set(test_df["customerID"])
assert train_ids.isdisjoint(valid_ids) and train_ids.isdisjoint(test_ids) and valid_ids.isdisjoint(test_ids)
print(f"Splits valides: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

# 4. Features restreintes pour le jalon 1
first_features = ["tenure", "MonthlyCharges"]
X_train = train_df[first_features].astype(float)
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

X_valid = valid_df[first_features].astype(float)
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

# 5. Modèle de référence (Baseline Dummy)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_valid)

print("\n--- BASELINE (Dummy Most Frequent) ---")
print("Accuracy:", accuracy_score(y_valid, dummy_preds))
print("Confusion Matrix (labels [0, 1]):\n", confusion_matrix(y_valid, dummy_preds, labels=[0, 1]))

# 6. Premier modèle entraîné (Régression Logistique)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_valid)

print("\n--- FIRST MODEL (Logistic Regression, 2 features) ---")
print("Accuracy:", accuracy_score(y_valid, preds))
print("Confusion Matrix (labels [0, 1]):\n", confusion_matrix(y_valid, preds, labels=[0, 1]))