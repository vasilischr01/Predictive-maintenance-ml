# Predictive Maintenance ML System

End-to-end machine learning system for predicting industrial machine failures using sensor and operational data.

The project covers the complete ML lifecycle: data validation, preprocessing, model training, threshold optimization, experiment tracking, explainability, API serving, drift monitoring, automated testing, and containerized deployment.

## Highlights

- Random Forest classifier for machine failure prediction
- Scikit-learn preprocessing and training pipeline
- Class imbalance handling with balanced class weights
- Decision-threshold analysis for precision/recall trade-offs
- MLflow experiment tracking and model logging
- FastAPI REST API for real-time inference
- SHAP-based prediction explanations
- Kolmogorov-Smirnov feature drift detection
- Dockerized deployment
- Automated API and data validation tests with pytest

## Model Performance

The selected operating threshold is `0.40`, providing a better balance between failure detection and false alarms than the default `0.50`.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9699 |
| Precision | 0.7313 |
| Recall | 0.7206 |
| F1 Score | 0.7259 |
| Decision Threshold | 0.40 |

The threshold was selected after evaluating multiple operating points and comparing precision, recall, F1 score, false positives, and false negatives.

## Architecture

```text
Raw Machine Data
        |
        v
Data Validation
        |
        v
Preprocessing Pipeline
  - Median imputation
  - Standard scaling
  - One-hot encoding
        |
        v
Random Forest Classifier
        |
        +--------------------+
        |                    |
        v                    v
   MLflow Tracking      Model Artifact
                             |
                             v
                       FastAPI Service
                        /health
                        /predict
                        /monitor/drift
                             |
                  +----------+----------+
                  |                     |
                  v                     v
          SHAP Explainability     Drift Monitoring
```

## Project Structure

```text
predictive-maintenance-ml/
├── artifacts/
│   ├── model.joblib
│   └── metrics.json
├── data/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── explainability/
│   │   └── explain.py
│   ├── monitoring/
│   │   └── drift.py
│   └── training/
│       ├── train.py
│       └── threshold_analysis.py
├── tests/
│   ├── test_api.py
│   └── test_validation.py
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

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
  "model_available": true
}
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

- predicted failure probability
- binary failure prediction
- operating threshold
- top SHAP feature contributions

### Drift Monitoring

```http
GET /monitor/drift
```

The monitoring endpoint compares feature distributions and reports potential distribution drift using the Kolmogorov-Smirnov test.

## MLflow Experiment Tracking

Model training runs are logged with MLflow, including:

- model parameters
- precision
- recall
- F1 score
- ROC-AUC
- class distribution
- decision threshold
- serialized model artifact

Start the MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

## Training

Train the model:

```bash
python -m src.training.train
```

Run threshold analysis:

```bash
python -m src.training.threshold_analysis
```

## Docker

Build the image:

```bash
docker build -t predictive-maintenance-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 predictive-maintenance-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Dockerized deployment has been validated for the health, prediction, and drift-monitoring endpoints.

## Testing

Run the automated test suite:

```bash
pytest -q
```

Current result:

```text
6 passed
```

The tests cover API behavior, request validation, drift monitoring, and dataset validation.

## Tech Stack

**Machine Learning**
- Python
- scikit-learn
- pandas
- NumPy

**MLOps / Explainability**
- MLflow
- SHAP
- SciPy

**Serving & Deployment**
- FastAPI
- Uvicorn
- Docker

**Testing**
- pytest

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset, containing industrial machine operating conditions and machine-failure labels.

The prediction features include:

- machine type
- air temperature
- process temperature
- rotational speed
- torque
- tool wear

## Design Decisions

### Class Imbalance

Machine failures are relatively rare, so the classifier uses balanced class weighting rather than optimizing only for overall accuracy.

### Threshold Optimization

A probability threshold of `0.40` is used instead of the default `0.50` to improve the balance between precision and recall for failure detection.

### Explainability

SHAP values provide feature-level explanations for individual predictions, making model decisions easier to inspect.

### Drift Detection

Feature distributions are monitored using two-sample Kolmogorov-Smirnov tests to identify statistically significant distribution changes.

### Reproducibility

Critical ML dependencies are pinned and the complete application can be executed inside a Docker container.

## Future Improvements

Potential extensions include:

- automated model retraining after drift detection
- CI/CD pipeline with GitHub Actions
- model registry and promotion workflow
- production metrics and alerting
- cloud deployment
- time-series-aware predictive maintenance models

## License

This project is intended for educational and portfolio purposes.