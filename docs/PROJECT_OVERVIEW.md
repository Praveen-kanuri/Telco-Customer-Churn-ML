# Telco Customer Churn ML — Project Overview

> Purpose of this file: a complete, factual description of what this repo contains, how it was built, and how it runs — written so it can be pasted into another LLM as source material to generate study notes (e.g. an ml-bible-style doc). It documents actual code, not aspirations.

---

## 1. Goal

Predict whether a telecom customer will churn (leave the service), and expose that prediction through:
- A trainable, tracked ML pipeline (MLflow + XGBoost)
- A REST API (FastAPI)
- A manual-testing web UI (Gradio, mounted inside the same FastAPI app)
- A Docker image deployable to AWS ECS Fargate behind an ALB

Dataset: `data/raw/Telco-Customer-Churn.csv` — the standard IBM/Kaggle Telco Churn dataset (~7,043 rows, one row per customer, target column `Churn` = Yes/No).

---

## 2. Repository layout

```
scripts/                    # entry points you run from the CLI
  run_pipeline.py           # THE training pipeline (load→validate→preprocess→features→train→log)
  prepare_processed_data.py # standalone helper to just produce data/processed/*.csv
  test_pipeline_phase1_data_features.py  # manual smoke test: load/validate/preprocess/features
  test_pipeline_phase2_modeling.py       # manual smoke test: train/evaluate
  test_fastapi.py           # manual smoke test: hits the running FastAPI /predict endpoint

src/
  data/
    load_data.py            # load_data(path) -> pd.read_csv with existence check
    preprocess.py            # preprocess_data(): strip headers, drop ID cols, Churn->0/1,
                              #   TotalCharges to numeric, SeniorCitizen to int, NaN->0 for numerics
  features/
    build_features.py        # build_features(): binary cols -> 0/1 (deterministic map),
                              #   multi-category cols -> pd.get_dummies(drop_first=True)
  model/
    train.py, evaluate.py, tune.py
                              # NOTE: standalone/legacy modules from earlier iteration.
                              # run_pipeline.py does NOT import these — it has its own inline
                              # training/eval code with different (tuned) hyperparameters.
                              # train.py uses n_estimators=300/lr=0.1/depth=6 and plain
                              # accuracy/recall; run_pipeline.py uses the optimized
                              # 301/0.034/7 config plus precision/recall/f1/roc_auc.
  utils/
    utils.py                 # small shared helpers
    validate_data.py          # validate_telco_data(): Great Expectations checks (see §5)
  serving/
    inference.py              # predict(): loads MLflow model + feature schema, transforms
                              #   a raw single-row dict identically to training, returns
                              #   "Likely to churn" / "Not likely to churn"
    model/                    # local copies of MLflow runs (for dev + copied into Docker image)
  app/
    main.py                   # FastAPI app + Gradio UI mounted at /ui — this is the app
                              #   actually used by CLAUDE.md's documented run commands and
                              #   the Dockerfile CMD.
    app.py                     # near-duplicate of main.py (same endpoints/UI), imports
                              #   `from serving.inference import predict` instead of
                              #   `from src.serving.inference import predict`, and has no
                              #   Pydantic docstrings/examples. Looks like an earlier version
                              #   kept alongside main.py rather than superseding it.

data/
  raw/                        # original CSV
  processed/                  # output of preprocess step (telco_churn_processed.csv)
  external/                   # (present, unused by pipeline code read so far)

artifacts/                    # feature_columns.json + preprocessing.pkl written by run_pipeline.py
                              # (local, dev-time mirror of what's also logged into MLflow)

mlruns/                       # MLflow's file-based tracking store (experiment "Telco Churn")

notebooks/
  EDA.ipynb                   # exploratory data analysis

configs/                      # present but empty — no config files currently checked in
great_expectations/           # present but effectively empty; GE is used purely in-code via
                              # ge.dataset.PandasDataset(df), not via a GE project/checkpoint setup

.github/workflows/ci.yml      # CI: build & push Docker image to Docker Hub on push to main
dockerfile                    # container build (see §7)
requirements.txt              # pinned dependency versions (see §8)
docs/ml-bible/                # separate, pre-existing set of condensed ML study notes
                              # (not touched by this file)
```

---

## 3. Training pipeline (`scripts/run_pipeline.py`)

Run with:
```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```
Optional flags: `--threshold` (default `0.35`), `--test_size` (default `0.2`), `--experiment` (default `"Telco Churn"`), `--mlflow_uri` (default: `{project_root}/mlruns` as a file URI).

Sequential stages, all inside one `mlflow.start_run()` context:

1. **Load** — `load_data(path)` reads the raw CSV.
2. **Validate** — `validate_telco_data(df)` runs a Great Expectations suite (see §5). Result logged as MLflow metric `data_quality_pass` (0/1). On failure, the failed expectation list is logged as an MLflow text artifact (`failed_expectations.json`) and the pipeline **raises and stops** — bad data never reaches training.
3. **Preprocess** — `preprocess_data(df)`: strip column names, drop ID columns, map `Churn` Yes/No → 1/0, coerce `TotalCharges` to numeric (raw data has blank strings for brand-new customers with `tenure == 0`), coerce `SeniorCitizen` to int, fill numeric NaNs with 0. Result written to `data/processed/telco_churn_processed.csv`.
4. **Feature engineering** — `build_features(df, target_col)`:
   - Binary categorical columns (exactly 2 unique values) → deterministic 0/1 mapping. Special-cased for `Yes/No` and `Male/Female`; any other 2-value column falls back to alphabetical-order mapping.
   - Multi-category columns (>2 unique values) → `pd.get_dummies(..., drop_first=True)`.
   - Any resulting boolean columns cast to int (XGBoost requires numeric input).
   - **Feature artifacts saved for serving-time consistency**: `artifacts/feature_columns.json` (local) and `feature_columns.txt` logged to MLflow — this exact column list/order is what `src/serving/inference.py` reindexes onto at serving time. `artifacts/preprocessing.pkl` (via joblib) bundles the feature column list + target name and is also logged to MLflow.
5. **Split** — stratified `train_test_split`, `test_size` from args, `random_state=42`.
6. **Class imbalance handling** — `scale_pos_weight = (# negative) / (# positive)` in the training set, passed to XGBoost so the minority (churn) class is upweighted.
7. **Train** — `XGBClassifier` with hardcoded, previously-tuned hyperparameters:
   `n_estimators=301, learning_rate=0.034, max_depth=7, subsample=0.95, colsample_bytree=0.98, eval_metric="logloss", scale_pos_weight=<computed>`. Training time logged as MLflow metric `train_time`.
8. **Evaluate** — predicts probabilities, applies the configurable classification `--threshold` (default 0.35 — lower than 0.5 to favor recall on churners), computes precision/recall/f1/roc_auc, all logged to MLflow. Inference time on the test set logged as `pred_time`.
9. **Log model** — `mlflow.sklearn.log_model(model, artifact_path="model")`, so the run's artifact directory ends up as `mlruns/<experiment_id>/<run_id>/artifacts/model/` (MLflow pyfunc-loadable format), alongside `feature_columns.txt` and `preprocessing.pkl`.

### Legacy modules not used by the pipeline
`src/model/train.py`, `evaluate.py`, `tune.py` implement an earlier/simpler version of training (untuned hyperparameters, accuracy/recall only, `mlflow.xgboost.log_model` instead of `mlflow.sklearn.log_model`). They are **not imported by `run_pipeline.py`** — worth knowing if another LLM tries to reconcile "the training code" and finds two versions.

---

## 4. Manual test scripts (no formal test suite)

CLAUDE.md and the code confirm there's no pytest-based suite; these are manual smoke-test scripts you run directly:
```bash
python scripts/test_pipeline_phase1_data_features.py   # load → validate → preprocess → features
python scripts/test_pipeline_phase2_modeling.py         # train → evaluate
python scripts/test_fastapi.py                          # exercises the running FastAPI /predict endpoint
```

---

## 5. Data validation (`src/utils/validate_data.py`)

Uses **Great Expectations 0.18** in "inline" mode — `ge.dataset.PandasDataset(df)` — not a full GE project (the `great_expectations/` directory in the repo is essentially empty/unused). Checks performed:
- Schema: `customerID`, `gender`, `Partner`, `Dependents`, `PhoneService`, `InternetService`, `Contract`, `tenure`, `MonthlyCharges`, `TotalCharges` must exist; `customerID` must not be null.
- Categorical value sets: `gender` ∈ {Male, Female}; `Partner`/`Dependents`/`PhoneService` ∈ {Yes, No}; `Contract` ∈ {Month-to-month, One year, Two year}; `InternetService` ∈ {DSL, Fiber optic, No}.
- Numeric ranges: `tenure` in [0, 120], `MonthlyCharges` in [0, 200], `TotalCharges` ≥ 0; `tenure`/`MonthlyCharges` not null.
- Cross-column consistency: `TotalCharges >= MonthlyCharges` for at least 95% of rows (`mostly=0.95`, allowing exceptions for edge cases/data entry noise).
- `TotalCharges` is coerced to numeric before validation (raw CSV has blank strings for zero-tenure customers).

Returns `(success: bool, failed_expectation_types: list[str])`, consumed by `run_pipeline.py` to gate training.

---

## 6. Serving (`src/app/main.py` + `src/serving/inference.py`)

Local run:
```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

- `GET /` — health check, `{"status": "ok"}` (this is the path the ALB target group health check hits in production).
- `POST /predict` — body validated against Pydantic model `CustomerData` (18 fields: demographics, phone/internet service flags, contract/billing/payment, plus numeric `tenure`/`MonthlyCharges`/`TotalCharges`). Calls `predict()` from `src/serving/inference.py`, returns `{"prediction": "Likely to churn" | "Not likely to churn"}` or `{"error": "..."}`.
- `/ui` — a `gradio.Interface` with one dropdown/number input per feature, mounted into the same FastAPI app via `gr.mount_gradio_app(app, demo, path="/ui")`. Includes two example rows (high-risk and low-risk profiles) and a themed UI (`gr.themes.Soft()`).

### `src/serving/inference.py` — the training/serving consistency layer
This is the module CLAUDE.md calls out as architecturally critical. On import:
1. Loads the MLflow pyfunc model from `MODEL_DIR = "/app/model"` (the container path). If that fails (e.g. local dev, no container), falls back to globbing `./mlruns/*/*/artifacts/model` and using whichever has the newest mtime.
2. Loads `feature_columns.txt` from the same directory as the resolved model — this defines `FEATURE_COLS`, the exact column order the model expects.

`predict(input_dict)`:
1. Wraps the single dict into a 1-row DataFrame.
2. `_serve_transform()` reproduces training-time feature engineering by hand, using a **hardcoded** `BINARY_MAP` (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` → fixed 0/1 mappings) rather than re-deriving it from data (since there's only one row, cardinality-based detection used at training time isn't possible at inference time).
3. Numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) coerced with `pd.to_numeric(errors="coerce")`, NaN→0.
4. Remaining object columns (multi-category) get `pd.get_dummies(drop_first=True)` — same call signature as training.
5. `df.reindex(columns=FEATURE_COLS, fill_value=0)` — forces the single-row frame into exactly the training column set/order; any one-hot column not produced by this single row (because most categories are absent) gets filled with 0, which is the correct "reference category" value.
6. Calls `model.predict()`, maps class `1` → `"Likely to churn"`, `0` → `"Not likely to churn"`.

### Duplicate app entry point
`src/app/app.py` is a near-duplicate of `main.py` — same endpoints and Gradio UI, but imports via `from serving.inference import predict` (relative to `src/` being on `sys.path`, vs. `main.py`'s `from src.serving.inference import predict`). CLAUDE.md documents both as alternative uvicorn targets, but the Dockerfile and the primary documented workflow use `main.py`.

---

## 7. Docker (`dockerfile`)

```dockerfile
FROM python:3.11-slim
# uv (fast pip) copied in from its own distroless image, used for dependency install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . .
COPY src/serving/model /app/src/serving/model
# Specific MLflow run's artifacts flattened to a convenience path the app loads from:
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key points:
- **The MLflow run ID (`3b1a41221fc44548aed629fa42b762e0`) is hardcoded** in the Dockerfile — every time a new "best" model is trained, this run ID must be manually updated (and that run's artifacts committed under `src/serving/model/`) before rebuilding the image. This is the mechanism by which "the model in production" is chosen; there's no automatic "promote latest/best run" step.
- `PYTHONPATH=/app/src` is what lets `src.app.main` / `src.serving.inference` imports resolve inside the container.
- Model artifacts are copied twice: once as the full nested MLflow run directory (`/app/src/serving/model/...`, for reference/local-dev parity) and once flattened to `/app/model` (what `inference.py`'s `MODEL_DIR` actually points at).

Build/run:
```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

---

## 8. CI/CD (`.github/workflows/ci.yml`)

- Trigger: push to `main`.
- Steps: checkout → Docker Buildx setup → Docker Hub login (`DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets) → build & push image tagged `<DOCKERHUB_USERNAME>/telco-fastapi:latest`.
- No test/lint step runs in CI — building/pushing the image is the only automated action.
- ECS deployment (forcing the running service to pick up the new image) is documented as a **manual** step (or a manual workflow trigger), not automated in this workflow file.

---

## 9. AWS deployment topology (per README; infra-as-code not present in repo)

- **ECS Fargate** runs the container (serverless, no EC2 management).
- **ALB** (Application Load Balancer) listens on HTTP:80, forwards to a **Target Group** using IP targets on HTTP:8000 (the container port).
- **Security groups**: ALB SG allows inbound 80 from `0.0.0.0/0`; task SG allows inbound 8000 *only* from the ALB SG.
- **Health checks**: ALB target group health check hits `GET /` on port 8000 (the FastAPI root endpoint).
- **CloudWatch Logs** capture container stdout/stderr and ECS service events.
- Traffic flow: push to `main` → CI builds/pushes image → ECS service force-redeployed (manual or scripted) → ALB health-checks new task → once healthy, users hit `POST /predict` or `/ui` via the ALB DNS name.

### Real issues hit and fixed (from README "Roadblocks")
1. **Unhealthy ALB targets** — app wasn't responding at the health-check path / listener-target port mismatch → fixed by adding `GET /`, aligning ALB listener (80) → target group (8000), setting health-check path to `/`.
2. **`ModuleNotFoundError: serving` in container** — `PYTHONPATH` didn't include `src/` → fixed by setting `PYTHONPATH=/app/src` in the Dockerfile and using `src.app.main:app` as the uvicorn target.
3. **ALB DNS timing out** — security group misconfiguration → fixed by scoping ALB SG (inbound 80 from internet) and task SG (inbound 8000 from ALB SG only).
4. **ECS redeploy not picking up new image** — service kept running the old task definition → fixed by forcing a new deployment after each image push.
5. **Gradio "No runs found in experiment"** — inference/UI expected a resolvable MLflow-logged model but couldn't find one → fixed by standardizing the MLflow experiment name and making model logging consistent, so inference always finds a run.
6. **Local vs. prod model path differences** — MLflow artifact URIs differ between a local `./mlruns/...` path and the container's flattened `/app/model` path → `inference.py`'s try/fallback logic (§6) is the resulting workaround.

---

## 10. MLflow tracking summary

- Tracking URI: file-based, `{project_root}/mlruns` (no tracking server).
- Experiment name: `"Telco Churn"` (overridable via `--experiment`).
- View locally: `mlflow ui --backend-store-uri file:./mlruns`.
- Per-run artifacts: `model/` (MLflow pyfunc), `feature_columns.txt`, `preprocessing.pkl`, and (on validation failure only) `failed_expectations.json`.
- Per-run params: `model` ("xgboost"), `threshold`, `test_size`.
- Per-run metrics: `data_quality_pass`, `train_time`, `pred_time`, `precision`, `recall`, `f1`, `roc_auc`.

---

## 11. Dependency stack (`requirements.txt` highlights)

ML/data: `pandas 2.1.4`, `numpy 1.26.4`, `scikit-learn 1.5.2`, `xgboost 3.0.3`, `lightgbm 4.6.0` (present but not used by the active pipeline), `optuna 4.4.0` (present, likely used by `src/model/tune.py`), `statsmodels`, `matplotlib`/`seaborn` (EDA).
Tracking/validation: `mlflow 2.14.1`, `great_expectations 0.18.19`.
Serving: `fastapi 0.115.0`, `uvicorn 0.30.5`, `gradio 4.44.1`, `pydantic 2.8.2`.
Infra/dev: `docker 7.1.0` (Python SDK), `pytest 8.3.2` (installed but no test suite uses it yet), `python-dotenv`.

---

## 12. Things worth flagging to a future reader / another LLM

- Two training implementations exist (`run_pipeline.py` inline vs. `src/model/train.py`); only the former is live.
- Two FastAPI+Gradio entry points exist (`src/app/main.py` vs `src/app/app.py`); only the former is used by Docker/CI.
- The production model is pinned by a hardcoded MLflow run ID inside the Dockerfile — promoting a new model is a manual edit + commit of new artifacts, not an automatic "latest" pointer.
- `configs/` and `great_expectations/` directories exist in the repo but are currently empty/unused — all validation logic and config are inline in Python, not externalized to files.
- No automated test suite (pytest is installed but unused) and no lint/test step in CI — correctness is verified by the manual `scripts/test_*.py` scripts.
- The current git branch (`feature/streamlit`) implies Streamlit work is planned/in-progress, but no Streamlit code exists in the repo yet.
