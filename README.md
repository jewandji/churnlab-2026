# IBM Telco Customer Churn - MLOps Foundation

## Business Problem and Target
The goal of this project is to identify telecommunications customers who are likely to cancel their contracts within the next 30 days.
- **Target column:** `Churn` (`Yes` / `No`).
- **Nature of the task:** Binary classification for retention targeting.
- **Dataset nuance:** This dataset is a single static customer snapshot, not an event-based transaction log or a randomized intervention study.

## Environment Setup
This project requires Python 3.11 managed via `uv`.

To recreate the isolated virtual environment deterministically:
```sh
uv sync --locked
```
To record current environment metadata:

```Bash
uv run python scripts/record_environment.py
```

**Data Provenance and Integrity Check**
The raw dataset is pinned to a specific commit from IBM's open repository.

Source URL: IBM Telco Customer Churn commit-pinned CSV

Expected SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91

Run the data validation and schema check:

```Bash
uv run python scripts/inspect_telco.py
```


**Key Directory Roles**
**src/churnlab/:** Core source code and package logic tracked directly in Git.

**data/raw/:** Read-only, immutable raw source snapshot validated via SHA-256 (not committed to Git).

**evidence/tp01/:** Reproducibility records, environment snapshots, and lab answers.

**reports/:** Automatically generated outputs and intermediate validation summaries.


### Part E - Collaboration Evidence
- **Pull Request URL:** https://github.com/jewandji/churnlab-2026/pull/1
- **Merge Commit Hash:** 2d85a02 (HEAD -> exercise/local-recovery, origin/main, main) Initialize Telco churn project with reproducible setup

### Part F - Recovery Exercise Explanation
- **Difference between F1 (`git restore <file>`) and F2 (`git restore --staged <file>`):**
  - `git restore <file>` modifies the working tree directly: it discards all unstaged local edits and reverts the file back to the state of the last commit. Any uncommitted work is permanently lost.
  - `git restore --staged <file>` only modifies the staging area (Git index): it un-stages changes prepared for the next commit, but leaves the physical modifications untouched in the working directory so they can be reviewed or edited further.