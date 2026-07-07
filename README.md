## Telco Churn – End-to-End ML Project
### Purpose

Build and ship a full machine-learning solution for predicting customer churn in a telecom setting—from data prep and modeling to an API + web UI deployed on AWS.

### Useful Commands

**Setup**
```bash
uv venv --python 3.11 .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv\Scripts\Activate.ps1 for PowerShell
uv pip install -r requirements.txt
```

**Run locally**
```bash
# FastAPI + Gradio UI (main entry point)
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Alternative entry point
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```
- API health check: `http://localhost:8000/`
- API docs (auto-generated): `http://localhost:8000/docs`
- Web UI: `http://localhost:8000/ui`

**Training pipeline**
```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
python scripts/prepare_processed_data.py   # prepare processed data only
```

**Testing**
```bash
python scripts/test_pipeline_phase1_data_features.py   # data processing + feature engineering
python scripts/test_pipeline_phase2_modeling.py        # model training + evaluation
python scripts/test_fastapi.py                         # FastAPI endpoints
```

**MLflow UI**
```bash
mlflow ui --backend-store-uri file:./mlruns
```

**Docker**
```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

**Troubleshooting: port 8000 already in use / wrong app responding**
```cmd
netstat -ano | findstr :8000
tasklist /FI "PID eq <pid>"
taskkill /PID <pid> /F
```
Always stop a running `uvicorn` with `Ctrl+C` in its terminal before starting a new one — leaving old instances running is what causes this.

### Problem solved & benefits

- Faster decisions: Predicts which customers are likely to churn so teams can act before they leave.
- Operationalized ML: Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- Repeatable delivery: CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- Traceable experiments: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST) and a root health check /.
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000; dependencies installed via `uv` for faster, more reliable builds.
- CI/CD: GitHub Actions builds the image and pushes to Docker Hub (tag resolved from the `DOCKERHUB_USERNAME` secret, not hardcoded); optionally triggers an ECS service update.
- Orchestration: AWS ECS Fargate runs the container (serverless).
- Networking: Application Load Balancer (ALB) on HTTP:80 forwarding to a Target Group (IP targets on HTTP:8000).
- Security: Security groups scoped to allow ALB inbound 80 from the internet, and task inbound 8000 from the ALB SG.
- Observability: CloudWatch Logs for container stdout/stderr and ECS service events in production; local runs also write structured logs to `logs/app.log` (every prediction request + result) via Python's `logging` module.

### Deployment flow (high-level)

- Push to main → GitHub Actions builds the Docker image and pushes it to Docker Hub.
- ECS service is updated (manually or via the workflow) to force a new deployment.
- ALB health checks hit / on port 8000; once healthy, traffic is routed to the new task.
- Users call POST /predict or open the Gradio UI at /ui via the ALB DNS.

### Roadblocks & how we solved them

Unhealthy targets behind ALB

- Cause: App didn’t respond at the health-check path; listener/target port mismatches.
- Fixes: Added GET / health endpoint; confirmed ALB listener on 80 forwards to TG on 8000; TG health check path set to /.

Module import error in container (ModuleNotFoundError: serving)

- Cause: Python path in the image didn’t include src/.
- Fixes: Set PYTHONPATH=/app/src in the Dockerfile; corrected uvicorn app path to src.app.main:app.

ALB DNS timing out

- Cause: Security group rules not aligned with traffic flow.
- Fixes: ALB SG allows inbound 80 from 0.0.0.0/0; task SG allows inbound 8000 from the ALB SG; outbound open.

ECS redeploy not picking up the new image

- Cause: Service still running previous task definition.
- Fixes: Force new deployment (CLI or console) after pushing the new image; optional step added to CI.

Gradio UI error (“No runs found in experiment”)

- Cause: Inference/UI expected an MLflow-logged model but couldn’t resolve a run.
- Fixes: Standardized MLflow experiment name and model logging in training; inference loads the logged model consistently (and a local path for dev).

Local testing vs. prod paths

- Cause: MLflow artifact URIs differ locally vs. in container.
- Fixes: For local dev, load via direct ./mlruns/.../artifacts/model; in prod, container loads the packaged model path used at build time.

Local fallback loaded the model but couldn't find feature_columns.txt

- Cause: The local-dev fallback in `inference.py` pointed `MODEL_DIR` at the mlflow `artifacts/model` folder itself, but `feature_columns.txt` actually lives one level up, as a sibling of `model/` under `artifacts/`.
- Fixes: Added a separate `FEATURE_DIR` variable that resolves correctly for both cases — same path as `MODEL_DIR` in Docker (where artifacts are flattened together), but the parent `artifacts/` folder in the local fallback path.

`uv pip install` failed with unsatisfiable dependency conflicts

- Cause: Switching from `pip` to `uv` for dependency installs exposed several pre-existing conflicts in `requirements.txt` that `pip` had silently tolerated — `ruamel.yaml==0.18.14` incompatible with `great_expectations`'s `<0.18` requirement, and an unpinned `gradio` resolving to `gradio==3.36.1` paired with an incompatible `gradio-client==2.5.0` (missing `gradio_client.serializing`), which cascaded into `pillow`, `MarkupSafe`, and `huggingface_hub` version conflicts.
- Fixes: Pinned `gradio==4.44.1` with matching compatible versions of its dependencies (`pillow==10.4.0`, `MarkupSafe==2.1.5`, `huggingface_hub==0.25.2`), and relaxed `ruamel.yaml` to `0.17.32`.

Docker Hub login failing in CI ("malformed HTTP Authorization header")

- Cause: `DOCKERHUB_TOKEN`/`DOCKERHUB_USERNAME` secret values had formatting issues (stray whitespace/newline from copy-paste), and the workflow's image tag was hardcoded to a stale account name instead of the repo's own Docker Hub username.
- Fixes: Regenerated a fresh Docker Hub access token, re-set both secrets cleanly, and changed the tag to `${{ secrets.DOCKERHUB_USERNAME }}/telco-fastapi:latest` so it always matches whoever's credentials are logged in.

Local UI returning 404 / wrong app entirely on localhost:8000

- Cause: A stale `uvicorn` process from a *different, unrelated* local project was still bound to `127.0.0.1:8000` from an earlier terminal session that was never stopped. Windows let that specific-IP bind coexist with this project's `0.0.0.0:8000` bind, so requests to `localhost:8000` were unpredictably routed to the wrong app.
- Fixes: `netstat -ano | findstr :8000` to find the PIDs holding the port, `tasklist`/`Get-CimInstance Win32_Process` to confirm what they actually were before touching anything, then `taskkill /PID <pid> /F` on the stale/orphaned one only. Going forward: always stop a running `uvicorn` with `Ctrl+C` in its terminal before starting a new one on the same port.