# Hospital Billing Anomaly Detection

AWS-deployed system for detecting anomalies in hospital billing data using a three-stage pipeline.

## Architecture

```
Hospital Billing Data
        |
        v
Stage 1: MLOps     ->  Data Preprocessing -> Anomaly Detection Model -> Structured Output
        |
        v
Stage 2: AgentOps  ->  LLM Insight Generation -> LLM Decision Support Agent
        |
        v
Stage 3: Decision  ->  Human-in-the-Loop -> Dashboard / Alerts -- this will be an LLM model which make the decision 
```

## Stage 1 - MLOps (Current Focus)

| Module              | Purpose                                              |
|---------------------|------------------------------------------------------|
| data_ingestion      | Ingest raw billing data from S3, validate schema     |
| data_preprocessing  | Clean, engineer features, prepare for modeling       |
| model_training      | Train Isolation Forest (unsupervised, no labels required)            |
| model_evaluation    | Threshold-based evaluation anchored to charge_amount_USD — anomaly score distribution, IQR/z-score thresholds, contamination rate analysis |
| model_registry      | Register approved models in SageMaker Model Registry |
| inference           | Score new records via SageMaker endpoint             |

## AWS Services Used

- S3 - Raw data, processed data, model artifacts
- SageMaker Pipelines - End-to-end training and inference pipelines
- SageMaker Model Registry - Model versioning and approval
- SageMaker Endpoints - Real-time inference
- CloudWatch - Monitoring and alerts

## Getting Started

```bash
pip install -r requirements/requirements-mlops.txt
cp .env.example .env
python pipelines/sagemaker_pipelines/training_pipeline.py
```

## Project Structure

```
src/stage_1_mlops/
    data_ingestion/       # S3 connector, schema, validator
    data_preprocessing/   # Cleaner, feature engineering, preprocessor
    model_training/       # Trainer, hyperparameter tuning, Isolation Forest
    model_evaluation/     # Threshold analyzer (charge_amount_USD), score distribution
    model_registry/       # SageMaker Model Registry integration
    inference/            # Predictor, anomaly scorer
```

<!-- pushed from yashwanth0098 -->
