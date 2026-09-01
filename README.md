# Predictive Maintenance ML System

[![CI](https://github.com/vasilischr01/Predictive-maintenance-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/vasilischr01/Predictive-maintenance-ml/actions/workflows/ci.yml)

End-to-end machine learning system for predicting industrial machine failures from operational and sensor data.

The project covers the complete ML lifecycle: data validation, leakage-free train/validation/test splitting, model benchmarking, decision-threshold optimization, experiment tracking, explainability, API serving, drift monitoring, automated testing, continuous integration, and reproducible Docker deployment.

## Highlights

- **Held-out test performance:** F1 **0.826**, ROC-AUC **0.982**, PR-AUC **0.909**, Recall **0.882**
- Leakage-free **70/15/15 stratified train/validation/test evaluation**
- Benchmark of **Logistic Regression, Random Forest, and HistGradientBoosting**
- Benchmark-driven production model selection using **validation PR-AUC**
- Validation-only decision-threshold optimization
- Final evaluation on an **untouched test set**
- HistGradientBoosting production classifier
- ROC-AUC and **PR-AUC** evaluation for imbalanced classification
- MLflow experiment tracking and model logging
- FastAPI REST API for real-time inference
- Cached SHAP-based prediction explanations
- Kolmogorov-Smirnov feature drift detection with practical effect-size filtering
- Controlled drift-injection experiment
- Reproducible Docker build that trains the production model during image creation
- Automated GitHub Actions CI
- API security headers and sanitized errors
- 64 KiB request-size limit
- 60 requests / 60 seconds / client-IP rate limiting
- **27 automated tests**

---

## Engineering Validation

```text
27 automated tests passed
Ruff: All checks passed
Security controls covered by API tests
Leakage-free train/validation/test evaluation
Docker build validated
```

---

## Final Model Performance

The production model is a `HistGradientBoostingClassifier`.

Model selection and threshold selection are performed exclusively on training and validation data. The test set is kept untouched until the final evaluation.

### Dataset Split

| Split | Rows | Fraction |
|---|---:|---:|
| Training | 7,000 | 70% |
| Validation | 1,500 | 15% |
| Test | 1,500 | 15% |

The positive machine-failure rate remains approximately 3.4% across all splits because stratified sampling is used.

### Selected Operating Point

The decision threshold is selected automatically on the validation set by maximizing F1 score.

```text
Selected threshold: 0.10
Selection split: validation
Candidate thresholds evaluated: 81
```

The threshold is persisted to:

```text
artifacts/selected_threshold.json
```

and loaded dynamically by the inference API.

### Untouched Test Performance

| Metric | Value |
|---|---:|
| Precision | **0.7759** |
| Recall | **0.8824** |
| F1 Score | **0.8257** |
| ROC-AUC | **0.9822** |
| PR-AUC | **0.9093** |
| Decision Threshold | **0.10** |

### Confusion Matrix

|  | Predicted Normal | Predicted Failure |
|---|---:|---:|
| Actual Normal | **1436** | **13** |
| Actual Failure | **6** | **45** |

The final operating point identifies **45 of 51 machine failures**, corresponding to **88.24% recall**, while producing 13 false-positive alerts.

---

## Model Benchmark

The benchmark compares:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Model selection uses **validation PR-AUC as the primary metric**, followed by validation F1 and recall as tie-breakers.

| Model | Selected Threshold | Validation Precision | Validation Recall | Validation F1 | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.88 | 0.3725 | 0.3725 | 0.3725 | 0.8402 | 0.3083 |
| Random Forest | 0.35 | 0.7115 | 0.7255 | 0.7184 | **0.9366** | 0.7164 |
| **HistGradientBoosting** | **0.10** | **0.7115** | **0.7255** | **0.7184** | 0.9295 | **0.7453** |

HistGradientBoosting achieved the highest validation PR-AUC and was selected for production.

Run the benchmark with:

```bash
python -m src.evaluation.model_benchmark
```

---

## Evaluation Methodology

```text
Full Dataset
    |
    v
Stratified Split
    |
    +--------------------+
    |                    |
    v                    v
70% Training        15% Validation
    |                    |
    |                    +--> Model comparison
    |                    +--> Threshold optimization
    |
    +-----------------------------+
                                  |
                                  v
                           Selection locked
                                  |
                                  v
                            15% Test Set
                                  |
                                  v
                         Final evaluation only
```

This prevents the test set from influencing model or threshold selection.

Metrics reported:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix

PR-AUC is emphasized because machine failures are rare.

---

## Architecture

```text
Raw Machine Data
      |
      v
Data Validation
      |
      v
Stratified Train / Validation / Test
      |
      +----------------------+----------------------+
      |                                             |
      v                                             v
Model Benchmark                              Training Pipeline
LR / RF / HGB                               Preprocessing + HGB
      |                                             |
      +----------------------+----------------------+
                             |
                             v
                  Validation Threshold Search
                             |
                             v
                  Production Model + Threshold
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
             MLflow                  FastAPI
                                         |
                              +----------+----------+
                              |                     |
                              v                     v
                            SHAP              Drift Monitoring
```

---

## Training

Run:

```bash
python -m src.training.train
```

The training pipeline:

1. validates the dataset
2. creates deterministic stratified train/validation/test splits
3. fits the HistGradientBoosting pipeline
4. evaluates candidate thresholds on validation data
5. selects the validation-optimal threshold
6. evaluates the locked model and threshold on the untouched test set
7. serializes the trained pipeline
8. persists metrics and threshold metadata
9. logs the experiment to MLflow

Generated artifacts include:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/selected_threshold.json
```

---

## Preprocessing Pipeline

### Categorical

- Machine type

### Numerical

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

Numerical preprocessing:

```text
Median Imputation
      |
      v
Standard Scaling
```

Categorical preprocessing:

```text
Most-Frequent Imputation
      |
      v
One-Hot Encoding
```

The preprocessing pipeline and classifier are serialized together.

---

## API

Start locally:

```bash
uvicorn src.api.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Health

```http
GET /health
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "type": "L",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}
```

The response includes:

- failure probability
- binary failure decision
- validation-selected threshold
- top SHAP feature contributions

### Drift Monitoring

```http
GET /monitor/drift
```

---

## API Security and Reliability

### Security Headers

Normal responses include:

```text
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy: camera=(), microphone=(), geolocation=()
Cache-Control: no-store
```

### Request Size Limit

Requests larger than **64 KiB** are rejected with:

```text
413 Request Entity Too Large
```

### Rate Limiting

```text
60 requests / 60 seconds / client IP
```

Exceeded limits return:

```text
429 Too Many Requests
Retry-After: 60
```

The current limiter is process-local and intended for single-instance/local deployment.

### Sanitized Errors

The API avoids returning raw internal exception strings.

Examples:

```text
503 Model artifacts are unavailable.
500 Prediction failed.
500 Prediction explanation failed.
503 Drift analysis artifacts are unavailable.
500 Drift analysis failed.
```

Internal details remain in server-side logs.

---

## Explainability

Predictions include SHAP feature-level explanations.

The explainability layer:

1. uses the fitted preprocessing pipeline
2. transforms the incoming observation
3. computes SHAP values
4. ranks features by absolute contribution magnitude
5. returns the top contributions

The model pipeline and SHAP explainer are cached after first use.

---

## Drift Monitoring

Each numerical feature is evaluated using:

- two-sample Kolmogorov-Smirnov statistic
- p-value
- practical KS effect-size threshold
- sample counts
- means
- mean shift
- standard deviations

A feature is flagged as drifted only when:

```text
p-value < 0.05
AND
KS statistic >= 0.08
```

### Controlled Drift Experiment

| Feature | KS Statistic | Drift |
|---|---:|---|
| Air temperature | 0.0284 | No |
| Process temperature | 0.0386 | No |
| Rotational speed | 0.0271 | No |
| **Torque** | **0.3106** | **Yes** |
| **Tool wear** | **0.1032** | **Yes** |

Run:

```bash
python -m src.monitoring.drift
```

---

## MLflow Experiment Tracking

Logged information includes:

- model type and hyperparameters
- random seed
- split proportions
- selected threshold
- validation metrics
- test metrics
- serialized model
- metrics artifact
- threshold artifact

Configuration is environment-driven through:

```text
MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME
```

---

## Docker

Build:

```bash
docker build -t predictive-maintenance-api .
```

Run:

```bash
docker run --rm -p 8000:8000 predictive-maintenance-api
```

The Docker build trains the production model during image creation, so a pre-generated model binary does not need to be committed.

---

## Testing

Run:

```bash
pytest -q
```

Current result:

```text
27 passed
```

Coverage includes:

### API

- health endpoint
- schema validation
- invalid machine types
- invalid operational values
- successful predictions
- drift endpoint behavior
- security headers
- oversized request rejection
- rate-limit enforcement
- sanitized model-artifact failures
- sanitized prediction failures
- sanitized SHAP failures
- sanitized drift failures

### Training / Evaluation

- deterministic split sizes
- class-proportion preservation
- threshold selection
- metric computation
- production classifier type

### Drift

- no-drift scenario
- controlled drift detection
- missing-feature rejection
- empty-batch rejection
- report persistence

### Validation

- valid dataset
- missing-column detection
- invalid machine-type detection

---

## Continuous Integration

GitHub Actions runs on:

```text
push
pull_request
```

The CI pipeline:

```text
Checkout
  |
  v
Python 3.11
  |
  v
Install Dependencies
  |
  v
Download Dataset
  |
  v
Compile Critical Modules
  |
  v
Run Tests
  |
  v
Build Docker Image
```

---

## Tech Stack

### Machine Learning

- Python
- scikit-learn
- pandas
- NumPy

### Evaluation

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Confusion matrices

### MLOps / Explainability

- MLflow
- SHAP
- SciPy

### Serving

- FastAPI
- Uvicorn
- Pydantic

### Monitoring

- Kolmogorov-Smirnov drift detection
- statistical-significance filtering
- practical effect-size filtering

### Deployment / Quality

- Docker
- pytest
- Ruff
- GitHub Actions

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

Production features:

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

Identifier fields and failure-mode indicator columns are excluded to avoid target-related leakage.

---

## Engineering Decisions

### Leakage-Free Evaluation

The test set is isolated before model or threshold selection.

### Imbalanced Classification

PR-AUC, recall, precision, F1, and ROC-AUC are reported instead of relying on accuracy.

### Benchmark-Driven Model Selection

HistGradientBoosting was selected because it achieved the highest validation PR-AUC.

### Threshold Optimization

Candidate thresholds from `0.10` to `0.90` are evaluated on validation data.

### Explainability

SHAP provides local feature-level explanations.

### Drift Detection

Statistical significance is combined with a minimum KS effect-size requirement.

### Reproducibility

The Docker image generates the model during build.

### Configuration and Secret Hygiene

MLflow configuration is environment-driven. The repository excludes `.env`, generated model artifacts, MLflow state, and raw data from version control.

---

## Limitations

- Single public dataset
- Statistical feature drift rather than full concept-drift monitoring
- Controlled synthetic drift demo
- No persistent production prediction database
- No automated retraining trigger
- Local/containerized deployment rather than managed cloud deployment
- No alert delivery or time-series dashboards
- Process-local rate limiter
- No authentication or RBAC layer

---

## Future Improvements

- Automated model retraining
- MLflow Model Registry
- Production prediction logging
- Drift-triggered retraining
- Concept-drift monitoring
- Cloud deployment
- Prometheus/OpenTelemetry metrics
- Dashboards and alerting
- Hyperparameter optimization
- Probability calibration
- Batch inference
- Authentication / RBAC
- Distributed rate limiting

---

## License

MIT License.
