# Predictive Maintenance ML System

[![CI](https://github.com/vasilischr01/Predictive-maintenance-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/vasilischr01/Predictive-maintenance-ml/actions/workflows/ci.yml)

End-to-end machine learning system for predicting industrial machine failures from operational and sensor data.

The project covers the complete ML lifecycle: data validation, leakage-free train/validation/test splitting, model benchmarking, decision-threshold optimization, experiment tracking, explainability, API serving, drift monitoring, automated testing, continuous integration, and reproducible Docker deployment.

## Highlights

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
- **19 automated tests**

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

It is not hardcoded into the serving layer.

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

| | Predicted Normal | Predicted Failure |
|---|---:|---:|
| Actual Normal | **1436** | **13** |
| Actual Failure | **6** | **45** |

The final operating point identifies **45 of 51 machine failures**, corresponding to a recall of **88.24%**, while producing 13 false-positive alerts.

---

## Model Benchmark

Rather than selecting a model a priori, the project evaluates several candidate classifiers using the same deterministic training and validation split.

The benchmark compares:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Model selection uses **validation PR-AUC as the primary metric**, followed by validation F1 and recall as tie-breakers.

The test set is not used during model selection.

| Model | Selected Threshold | Validation Precision | Validation Recall | Validation F1 | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.88 | 0.3725 | 0.3725 | 0.3725 | 0.8402 | 0.3083 |
| Random Forest | 0.35 | 0.7115 | 0.7255 | 0.7184 | **0.9366** | 0.7164 |
| **HistGradientBoosting** | **0.10** | **0.7115** | **0.7255** | **0.7184** | 0.9295 | **0.7453** |

HistGradientBoosting achieved the highest validation PR-AUC and was therefore promoted to the production model.

The complete benchmark is written to:

```text
artifacts/model_benchmark.json
```

Run the benchmark with:

```bash
python -m src.evaluation.model_benchmark
```

---

## Evaluation Methodology

The project explicitly separates model training, model selection, threshold selection, and final evaluation.

```text
Full Dataset
     |
     v
Stratified Split
     |
     +--------------------+
     |                    |
     v                    v
70% Training         15% Validation
     |                    |
     |                    +--> Model comparison
     |                    |
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

This prevents the test set from influencing either:

- model selection
- threshold selection

and provides a more reliable estimate of final model performance.

For the imbalanced machine-failure target, the evaluation reports:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix

PR-AUC is included because machine failures represent only a small fraction of the dataset.

---

## Architecture

```text
                         Raw Machine Data
                                |
                                v
                         Data Validation
                                |
                                v
                  Stratified Train / Val / Test
                                |
                +---------------+---------------+
                |                               |
                v                               v
         Model Benchmark                Training Pipeline
   Logistic Regression                  - Median imputation
   Random Forest                        - Standard scaling
   HistGradientBoosting                 - One-hot encoding
                |                               |
                v                               v
      Validation Model Selection    HistGradientBoosting
                |                               |
                +---------------+---------------+
                                |
                                v
                   Validation Threshold Search
                                |
                                v
                  Production Model + Threshold
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
        MLflow Tracking                     FastAPI Service
                                              /health
                                              /predict
                                              /monitor/drift
                                                   |
                           +-----------------------+----------------------+
                           |                                              |
                           v                                              v
                   SHAP Explainability                            Drift Monitoring
```

---

## Project Structure

```text
predictive-maintenance-ml/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── artifacts/
│   ├── model.joblib
│   ├── metrics.json
│   ├── selected_threshold.json
│   ├── model_benchmark.json
│   └── drift_report.json
│
├── data/
│   └── raw/
│       └── ai4i2020.csv
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── data/
│   │   ├── download.py
│   │   └── validate.py
│   │
│   ├── evaluation/
│   │   └── model_benchmark.py
│   │
│   ├── explainability/
│   │   └── explain.py
│   │
│   ├── monitoring/
│   │   └── drift.py
│   │
│   └── training/
│       ├── train.py
│       └── threshold_analysis.py
│
├── tests/
│   ├── test_api.py
│   ├── test_drift.py
│   ├── test_training.py
│   └── test_validation.py
│
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Generated model and evaluation artifacts are intentionally kept out of version control and can be reproduced by the training pipeline.

---

## Training

Train the production model:

```bash
python -m src.training.train
```

The training pipeline:

1. validates the dataset,
2. creates deterministic stratified train/validation/test splits,
3. fits the production HistGradientBoosting pipeline,
4. evaluates candidate thresholds on validation data,
5. selects the validation-optimal threshold,
6. evaluates the locked model and threshold on the untouched test set,
7. serializes the trained pipeline,
8. persists metrics and threshold metadata,
9. logs the experiment to MLflow.

Generated artifacts include:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/selected_threshold.json
```

---

## Preprocessing Pipeline

The model receives the following input features.

### Categorical

- Machine type

### Numerical

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The scikit-learn pipeline performs:

### Numerical preprocessing

```text
Median Imputation
        |
        v
Standard Scaling
```

### Categorical preprocessing

```text
Most-Frequent Imputation
        |
        v
One-Hot Encoding
```

The fitted preprocessing pipeline and classifier are serialized together, ensuring that training and inference use identical transformations.

---

## API

Start the API locally:

```bash
uvicorn src.api.main:app --reload
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_available": true,
  "threshold_available": true
}
```

The health endpoint checks the availability of both the trained model and validation-selected threshold artifact.

---

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

The response contains:

- failure probability
- binary failure decision
- validation-selected operating threshold
- top SHAP feature contributions

The prediction decision is calculated as:

```text
predicted_failure =
    failure_probability >= selected_threshold
```

The threshold is loaded from `artifacts/selected_threshold.json` rather than being hardcoded into the API.

---

## Explainability

Predictions include feature-level explanations generated with SHAP.

The explainability layer:

1. uses the fitted preprocessing pipeline,
2. transforms the incoming observation,
3. computes SHAP values for the HistGradientBoosting classifier,
4. ranks features by absolute contribution magnitude,
5. returns the top contributions with the prediction.

The model pipeline and SHAP explainer are cached after first use to avoid repeatedly loading the serialized model and rebuilding the explainer for every API request.

Example explanation structure:

```json
[
  {
    "feature": "Torque",
    "shap_value": 0.42
  }
]
```

Positive and negative SHAP values describe how individual transformed features influence the model output relative to its baseline prediction.

---

## Drift Monitoring

```http
GET /monitor/drift
```

The monitoring module supports direct comparison between:

```text
Reference Batch
      |
      v
Two-Sample KS Test
      ^
      |
Current Batch
```

Each numerical feature is evaluated using:

- two-sample Kolmogorov-Smirnov statistic
- p-value
- practical KS effect-size threshold
- reference and current sample counts
- reference and current means
- mean shift
- reference and current standard deviations

A feature is classified as drifted only when:

```text
p-value < 0.05
AND
KS statistic >= 0.08
```

Using both statistical significance and a practical effect-size threshold avoids flagging very small distribution differences purely because of large sample sizes.

### Controlled Drift Experiment

The repository also contains a deterministic drift-injection scenario for demonstration and testing.

Reference and current batches are created from randomized, disjoint samples of the same source distribution. Controlled distribution shifts are then injected into:

- `Torque`
- `Tool wear`

The detector correctly identifies:

```text
Torque      -> drift detected
Tool wear   -> drift detected
```

while leaving:

```text
Air temperature      -> no drift
Process temperature  -> no drift
Rotational speed     -> no drift
```

Observed KS statistics in the controlled experiment:

| Feature | KS Statistic | Drift |
|---|---:|---|
| Air temperature | 0.0284 | No |
| Process temperature | 0.0386 | No |
| Rotational speed | 0.0271 | No |
| **Torque** | **0.3106** | **Yes** |
| **Tool wear** | **0.1032** | **Yes** |

The synthetic scenario is intended only for reproducible demonstration and testing.

For production use, the same drift functions can accept real reference and current production batches directly.

Run the demo with:

```bash
python -m src.monitoring.drift
```

The generated report is written to:

```text
artifacts/drift_report.json
```

---

## MLflow Experiment Tracking

Training runs are tracked with MLflow.

Logged information includes:

- model type
- model hyperparameters
- random seed
- dataset split proportions
- selected decision threshold
- threshold selection metric
- validation precision
- validation recall
- validation F1
- validation ROC-AUC
- validation PR-AUC
- test precision
- test recall
- test F1
- test ROC-AUC
- test PR-AUC
- serialized model
- metrics artifact
- threshold artifact

The local tracking URI can be configured through:

```text
MLFLOW_TRACKING_URI
```

and the experiment name through:

```text
MLFLOW_EXPERIMENT_NAME
```

For example, a local MLflow server can be started with:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

and training can then be executed with:

```bash
python -m src.training.train
```

Docker builds use a local SQLite MLflow backend so that model generation does not depend on an external tracking server.

---

## Docker

The Docker image is fully reproducible from source.

The build process:

```text
Python 3.11 Base Image
        |
        v
Install Dependencies
        |
        v
Copy Source + Dataset
        |
        v
Train Production Model
        |
        +--> model.joblib
        +--> metrics.json
        +--> selected_threshold.json
        |
        v
Start FastAPI
```

The repository therefore does not require a pre-generated model binary to build the production image.

### Build

```bash
docker build -t predictive-maintenance-api .
```

The Docker build has been validated successfully with the production training pipeline.

### Run

```bash
docker run --rm -p 8000:8000 predictive-maintenance-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Validated container response:

```json
{
  "status": "ok",
  "model_available": true,
  "threshold_available": true
}
```

---

## Testing

Run the complete test suite:

```bash
pytest -q
```

Current result:

```text
19 passed
```

The suite covers:

### API

- health endpoint
- schema validation
- invalid machine types
- invalid operational values
- successful high-risk predictions
- below-threshold predictions
- drift endpoint behavior

### Training and Evaluation

- deterministic train/validation/test split sizes
- preservation of class proportions
- threshold selection
- metric computation
- production classifier type

### Drift Monitoring

- no-drift scenario
- controlled drift detection
- missing feature rejection
- empty-batch rejection
- drift report persistence

### Dataset Validation

- valid input dataset
- missing-column detection
- invalid machine-type detection

---

## Continuous Integration

GitHub Actions runs automatically on:

```text
push
pull_request
```

The CI pipeline performs:

```text
Checkout
   |
   v
Python 3.11 Setup
   |
   v
Install Dependencies
   |
   v
Download Dataset
   |
   v
Compile Critical Python Modules
   |
   v
Run 19 Tests
   |
   v
Build Docker Image
```

This verifies both application correctness and reproducible containerization on a clean CI environment.

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
- Validation-based model selection
- Validation-based threshold optimization

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

### Deployment

- Docker

### Testing / CI

- pytest
- GitHub Actions

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

It contains industrial machine operating conditions and machine-failure labels.

The production model uses:

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

as predictive features.

Identifier fields and failure-mode indicator columns are not used as model inputs.

This avoids introducing target-related leakage into the feature set.

---

## Engineering Decisions

### Leakage-Free Evaluation

The test set is isolated before model or threshold selection.

Model comparison occurs on validation data, and the final test set is evaluated only after the model and operating threshold have been locked.

### Imbalanced Classification

Machine failures represent only a small fraction of observations.

For this reason, evaluation does not rely on accuracy alone.

PR-AUC, recall, precision and F1 are reported alongside ROC-AUC.

### Benchmark-Driven Model Selection

The production model was not chosen arbitrarily.

Logistic Regression, Random Forest and HistGradientBoosting were evaluated under the same train/validation split.

HistGradientBoosting achieved the highest validation PR-AUC and was selected for production.

### Threshold Optimization

The default probability threshold of `0.50` is not assumed to be optimal for failure detection.

Candidate thresholds from `0.10` to `0.90` are evaluated on validation data.

The selected threshold is persisted as an artifact and reused by the API.

### Explainability

SHAP provides local feature-level explanations for individual predictions.

The explainer is cached to avoid unnecessary model loading and initialization on every request.

### Drift Detection

Drift monitoring combines statistical significance with a minimum KS effect-size requirement.

This reduces the risk of treating statistically detectable but operationally negligible distribution changes as meaningful drift.

### Reproducibility

The Docker image generates the production model during the image build.

This ensures the application can be reproduced from source without requiring a locally generated model binary to be committed to Git.

### Configuration

MLflow configuration is environment-driven rather than hardcoded.

This allows the same training pipeline to run against different tracking backends in local development, CI, or containerized environments.

---

## Limitations

This project is a portfolio-scale ML system rather than a live industrial deployment.

Current limitations include:

- training and evaluation use a single public dataset
- drift monitoring is statistical feature monitoring rather than full concept-drift detection
- the included drift demonstration uses controlled synthetic distribution shifts
- there is no persistent production prediction database
- there is no automated retraining trigger
- the API is currently deployed locally/containerized rather than to a managed cloud service
- monitoring does not yet include alert delivery or time-series dashboards

These limitations are intentionally documented to distinguish demonstrated engineering functionality from infrastructure that would be required in a real production environment.

---

## Future Improvements

- Automated model retraining workflow
- MLflow Model Registry integration
- Production prediction logging
- Drift-triggered retraining policies
- Concept-drift monitoring
- Cloud deployment
- Prometheus/OpenTelemetry metrics
- Monitoring dashboards
- Alerting
- Hyperparameter optimization
- Probability calibration
- Additional ensemble models
- Batch inference endpoint
- Authentication and rate limiting

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.