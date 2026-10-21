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