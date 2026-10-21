import joblib
import pandas as pd
from churnlab.pipeline import parse_inputs

# 1. Chargement de l'artefact sérialisé
pipeline = joblib.load("artifacts/churn_pipeline.joblib")

# 2. Échantillon de nouveaux clients avec les types d'origine
new_customers = pd.DataFrame([
    {
        "tenure": 2,
        "MonthlyCharges": 70.35,
        "TotalCharges": "140.7",
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    },
    {
        "tenure": 45,
        "MonthlyCharges": 20.0,
        "TotalCharges": "900.0",
        "Contract": "One year",
        "InternetService": "No",
        "PaymentMethod": "Mailed check",
    },
])

# 3. Application du parsing sans état puis inférence directe
clean_data = parse_inputs(new_customers)
predictions = pipeline.predict(clean_data)
probabilities = pipeline.predict_proba(clean_data)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
    status = "CHURN" if pred == 1 else "RETAIN"
    print(f"Client {i} -> Décision : {status} (Probabilité : {prob:.4f})")