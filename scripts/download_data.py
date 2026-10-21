import hashlib
import json
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
csv_path = raw_dir / "telco.csv"
meta_path = raw_dir / "telco.source.json"

print(f"Téléchargement depuis {DATA_URL}...")
urllib.request.urlretrieve(DATA_URL, csv_path)

actual_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise ValueError(f"Hash invalide: attendu {EXPECTED_SHA256}, obtenu {actual_sha256}")

meta = {
    "url": DATA_URL,
    "sha256": actual_sha256,
}
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Téléchargement réussi et hash SHA-256 vérifié : {actual_sha256}")