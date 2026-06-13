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

## Data Description

| Column Name | Description | Example Values |
|---|---|---|
| patient_id | Unique identifier assigned to each patient | P12345 |
| gender | Gender of the patient | Male, Female, Other |
| age | Age of the patient (in years) | 35, 62 |
| visit_date | Date when the patient visited the hospital/clinic | 2023-07-14 |
| department | Hospital department where the patient was seen | Cardiology, Oncology, Emergency |
| physician_id | Unique identifier for the attending physician | DR4567 |
| diagnosis | Primary diagnosis assigned during the visit (ICD code or text) | Hypertension, Diabetes Mellitus |
| visit_type | Type of visit | Outpatient, Inpatient, Emergency |
| visit_reason | Reason for the visit as stated by the patient or recorded | Chest pain, Routine checkup |
| appointment_id | Unique identifier for the appointment (helps detect duplicate or multiple claims) | A78901 |
| is_emergency | Indicates if the visit was marked as an emergency | Yes/No or 1/0 |
| insurance_id | Unique identifier of the patient's insurance provider | INS8765 |
| payer_name | Name of the payer/insurance company responsible for covering charges | Medicare, BlueCross, Private |
| payer_type | Type of payer (public, private, self-pay, government) | Private, Government, Self |
| claim_status | Current status of the insurance claim | Pending, Approved, Rejected, Paid |
| charge_amount_USD | Total charges billed for the visit (in US Dollars) — **primary anomaly reference column** | 500.00, 3500.75 |
| payment_amount_USD | Actual payment received from payer/patient (in US Dollars) | 450.00, 0.00 |
| admission_date | Date of hospital admission (for inpatients) | 2023-07-12 |
| discharge_date | Date of hospital discharge (for inpatients) | 2023-07-15 |

> `charge_amount_USD` is the primary reference column used for threshold-based anomaly evaluation in Stage 1.

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
