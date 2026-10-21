### Part A - Understand the project and data

**Question A1:** Why should the original CSV, a fitted model and the training source code have different roles in the repository?

These three elements have fundamentally distinct lifecycles, formats, and traceability requirements within a machine learning system. The training source code (src/, scripts/) consists of lightweight, editable text instructions; it is managed directly by Git to track the history of changes, facilitate collaboration, and automate pipelines. In contrast, the original CSV file (data/raw/) is a raw, immutable snapshot that serves as the input reference: it must never be modified in place, and to avoid unnecessarily bloating the code repository, its provenance is tracked using a cryptographic fingerprint (SHA-256 hash) or dedicated data storage rather than committing it with every revision. Finally, the trained model (artifacts/) is a large binary artifact resulting from the deterministic execution of a job that combines code, data, configuration, and environment ($\text{Modèle} = f(\text{Code}, \text{Data}, \text{Config}, \text{Env})$). Since it can be regenerated from these components and requires specific monitoring (evaluation metrics, governance, and deployment), it belongs in a model registry or artifact repository, separate from the source code. 



**Question A2:** Why would including Churn among the model inputs invalidate the prediction task?

Including the “Churn” column among the explanatory variables would cause significant target leakage, destroying the model’s mathematical and operational validity. The Churn value corresponds to the exact label that the algorithm is supposed to infer from customer behavior: if the model has access to it directly from the input features, it will learn a trivial rule with no predictive value ($\text{prédiction} = \text{entrée}$) rather than identifying actual signals related to tenure, billing, or platform. In practical terms, this variable is, by definition, unavailable at the time of weekly inference regarding active customers; faced with the absence of this information in production data streams, the system would be unable to identify profiles that are about to cancel their subscriptions.


Here are the answers for parts **B** and **C**, written in student-level English as separate paragraphs for your `evidence/tp01/answers.md` file.


### Part B - Environment and dependencies

#### Task B2.1 & B2.2: Python constraint and direct dependency

In `pyproject.toml`, project requirements are specified in a human-readable and declarative way. The Python interpreter constraint is defined under `requires-python = ">=3.11,<3.13"`, and top-level dependencies are listed in the `dependencies` array, such as `pandas>=2.2.0`. In `uv.lock`, this same dependency is pinned inside a dedicated `[[package]]` block that records its exact resolved version (for instance `3.0.6` or `2.2.x`), its complete list of required sub-dependencies, and cryptographic distribution hashes for each source archive and wheel. This lockfile ensures that any installer pulls the exact same artifacts without recalculating dependency versions.

#### Task B2.3: Transitive dependency with `uv tree`

Running `uv tree` reveals the project's dependency hierarchy and clearly exposes transitive dependencies. A transitive dependency is a package not directly listed in our `pyproject.toml`, but pulled in because a direct dependency relies on it. For instance, `joblib`, `scipy`, and `threadpoolctl` are shown nested underneath `scikit-learn`, while `python-dateutil` and `six` appear underneath `pandas`. These packages were automatically resolved and installed by `uv` to satisfy the requirements of our primary libraries.

#### Task B2.4: Difference between `pyproject.toml`, `uv.lock`, and `.venv/`

These three elements handle environment management at different abstraction layers. `pyproject.toml` is the high-level specification written for humans to declare version bounds, project metadata, and primary requirements. `uv.lock` is the deterministic, machine-readable dependency graph that pins the exact version and hash of every single installed package across environments. Finally, `.venv/` is the physical runtime directory on the local machine containing the concrete Python binaries, compiled C extensions, and installed site-packages files for that specific operating system.

#### Task B3: Version specifier interpretation

The version constraint `numpy ~= 1.26.0` uses the compatible release operator, which is equivalent to the range `>=1.26.0, <1.27.0`. This allows bugfix and patch releases while blocking breaking minor version bumps. In contrast, an exact pin like `numpy==1.26.4` restricts the environment to a single build. Specifier ranges leave room for the resolver to pick the best match, whereas a lockfile freezes that choice permanently. Even so, adhering to semantic versioning alone does not guarantee full cross-platform compatibility across distinct hardware architectures.

---

### Part C - Telco dataset inspection and fingerprinting

#### Question C3.1: Why churn prediction is a classification task

Predicting customer churn is a supervised classification task because the target variable (`Churn`) is discrete and categorical, taking only two mutually exclusive states: `Yes` or `No`. Unlike a regression problem where the model outputs a continuous numeric quantity (such as monthly revenue or customer lifetime), churn modeling estimates the probability that an active user will belong to the departure class so retention teams can prioritize interventions.

#### Question C3.2: Class balance and the limitation of accuracy

The Telco dataset has an imbalanced target distribution: retained customers (`Churn = No`) account for approximately 73.46% (5,174 rows), while churning customers (`Churn = Yes`) make up only 26.54% (1,869 rows). In this setting, raw accuracy is a misleading metric because a dummy model that blindly predicts `No` for everyone would reach 73.5% accuracy while failing to detect a single leaving customer. For production decisions, evaluation must rely on threshold-independent or business-aligned metrics such as ROC-AUC, PR-AUC, recall, or precision at the top 5% contact capacity.

#### Question C3.3: Why `TotalCharges` is loaded as an object/string

Although `TotalCharges` conceptually represents a continuous monetary amount, Pandas parses the column as an `object` (string) type by default. This happens because the raw CSV contains non-numeric text values, specifically empty whitespace strings (`" "`) present in 11 rows corresponding to new customers with zero tenure (`tenure = 0`). Because Pandas encounters a mix of valid floats and whitespace strings during initial schema inference, it falls back to the most general type (`object`) to prevent data loss.

#### Question C3.4: Why `isna()` misses whitespace-only values

The `.isna()` method in Pandas only flags recognized null sentinels, such as `np.nan`, `None`, or `pd.NA`. A cell containing spaces (`" "`) is considered a valid, non-empty Python string with a length greater than zero, so `.isna()` evaluates to `False`. Detecting these hidden missing entries requires stripping whitespace first (e.g., using `.str.strip().eq("")`) or explicitly coercing the column with `pd.to_numeric(df["TotalCharges"], errors="coerce")`, which converts unparseable strings into real `NaN` values.

#### Question C3.5: Verifiable evidence of identical raw data

Sharing a filename like `telco.csv` or comparing row counts is not enough to prove data integrity, as two different files can share identical names or row totals while having conflicting values. The only tamper-proof evidence is a deterministic SHA-256 cryptographic hash computed over the raw bytes of the file (`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`). Matching this exact hash guarantees bit-level parity between two machines.