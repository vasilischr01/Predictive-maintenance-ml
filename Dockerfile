FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV MLFLOW_EXPERIMENT_NAME=predictive-maintenance

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

RUN mkdir -p artifacts

# Reproducibly generate the production model and threshold artifacts.
RUN python -m src.training.train

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]