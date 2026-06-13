# Hospital Billing Anomaly Detection

## Problem Statement

Hospitals generate large amounts of structured data from patient visits, admissions, and billing transactions. While most records follow expected clinical and financial patterns, some may represent anomalies such as incorrect charges, fraudulent claims, unusual patient-diagnosis mismatches, or abnormal physician billing behavior.

Detecting these anomalies manually is time-consuming, error-prone, and does not scale with the volume of modern hospital operations. There is a need for an automated, intelligent system that can continuously monitor billing data, flag suspicious records, generate human-readable insights, and support decision-makers in taking corrective action — all without requiring pre-labeled training data.

This project addresses that need through a three-stage AWS-deployed pipeline:

- **Stage 1 — MLOps:** Unsupervised anomaly detection using Isolation Forest on raw billing records, with threshold-based scoring anchored to `charge_amount_USD`, fully orchestrated via Amazon SageMaker Pipelines.
- **Stage 2 — AgentOps:** LLM-based insight generation (Amazon Bedrock) that explains each detected anomaly in context, backed by a decision-support agent with access to billing policy tools.
- **Stage 3 — Decision LLM:** An LLM model that makes the final decision — flag, escalate, or close — feeding results to a human-review dashboard and alert system.

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
