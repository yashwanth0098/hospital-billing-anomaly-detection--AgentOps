"""
dev_run.py - Stage 1 MLOps: Data Ingestion + Validation + Drift Detection

Run from the project root:
    python dev_run.py
"""
import copy
import io
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# ── stdout / stderr: force UTF-8 so Unicode chars render on Windows ───────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── loguru: remove default handler, add console + file sinks ─────────────────
logger.remove()

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
    "{name}:{function}:{line} — {message}"
)

# Console sink — INFO and above, write to stdout so logs interleave with print()
logger.add(
    sys.stdout,
    format=_LOG_FORMAT,
    level="INFO",
    colorize=False,   # Windows console colours are unreliable; keep it plain
)

# File sink — DEBUG and above (full trace for every run)
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/pipeline.log",
    format=_LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",    # new file after 10 MB
    retention="7 days",  # keep one week of logs
    encoding="utf-8",
)

from src.stage_1_mlops.data_ingestion import DataIngestionPipeline
from src.stage_1_mlops.data_ingestion.data_validator import DataValidator


# ── helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    """Print a passing check line."""
    print(f"  [PASS]  {msg}")


def fail(msg: str) -> None:
    """Print a failing check line and exit."""
    print(f"  [FAIL]  {msg}")
    sys.exit(1)


# ── Stage 1: run the full pipeline ───────────────────────────────────────────

section("STAGE 1 - DataIngestionPipeline.run()")

pipeline = DataIngestionPipeline()
try:
    df = pipeline.run()
    ok(f"Loaded and cleaned : {df.shape[0]} rows x {df.shape[1]} columns")
except (ValueError, FileNotFoundError, OSError) as e:
    fail(str(e))


# ── Stage 2: shape and dtypes ────────────────────────────────────────────────

section("STAGE 2 - Column dtypes after cleaning")
print(df.dtypes.to_string())


# ── Stage 3: null counts ─────────────────────────────────────────────────────

section("STAGE 3 - Null counts after cleaning")
nulls = df.isnull().sum()
if nulls.sum() == 0:
    ok("No nulls remaining in any column.")
else:
    print(nulls[nulls > 0].to_string())


# ── Stage 4: date logic sanity ────────────────────────────────────────────────

section("STAGE 4 - Date logic sanity checks")

outpatient_with_dates = df[(df["visit_type"] == "Outpatient") & df["admission_date"].notna()]
inpatient_without_dates = df[(df["visit_type"] == "Inpatient") & df["admission_date"].isna()]

if outpatient_with_dates.empty:
    ok("Outpatient rows: no spurious admission/discharge dates  (0 rows)")
else:
    fail(f"Outpatient rows have admission dates -- {len(outpatient_with_dates)} rows affected!")

if inpatient_without_dates.empty:
    ok("Inpatient rows: all have admission + discharge dates  (0 missing)")
else:
    fail(f"Inpatient rows are missing dates -- {len(inpatient_without_dates)} rows affected!")

bad_dates = df[
    df["admission_date"].notna()
    & df["discharge_date"].notna()
    & (df["discharge_date"] < df["admission_date"])
]
if bad_dates.empty:
    ok("No rows where discharge_date < admission_date")
else:
    fail(f"{len(bad_dates)} rows have discharge before admission!")


# ── Stage 5: categorical value distributions ─────────────────────────────────

section("STAGE 5 - Categorical distributions")

categoricals = [
    "gender", "visit_type", "department",
    "payer_type", "claim_status", "is_emergency",
]

for col in categoricals:
    counts = df[col].value_counts().to_dict()
    print(f"\n  {col}:")
    for val, cnt in sorted(counts.items()):
        bar = "#" * (cnt // 100)
        print(f"    {val:<30} {cnt:>5}  {bar}")


# ── Stage 6: numeric summary ─────────────────────────────────────────────────

section("STAGE 6 - Numeric column summary")
print(df[["age", "charge_amount_USD", "payment_amount_USD"]].describe().round(2).to_string())


# ── Stage 7: validator edge-case smoke tests ─────────────────────────────────

section("STAGE 7 - Validator smoke tests (edge cases)")

# Suppress loguru during smoke tests — failures here are intentional.
logger.disable("src.stage_1_mlops.data_ingestion.data_validator")

v = DataValidator()

_base = {
    "patient_id": [1], "gender": ["Male"], "age": [30],
    "visit_date": ["01-01-2024"], "department": ["Cardiology"],
    "physician_id": [100], "diagnosis": ["Flu"], "visit_type": ["Outpatient"],
    "visit_reason": ["Checkup"], "appointment_id": ["A00001"],
    "is_emergency": ["No"], "insurance_id": ["INS-ABC123"],
    "payer_name": ["Acme"], "payer_type": ["Insurance"],
    "claim_status": ["Submitted"], "charge_amount_USD": [200],
    "payment_amount_USD": [0], "admission_date": [None],
    "discharge_date": [None],
}


def make(**overrides) -> pd.DataFrame:
    """Return a one-row test DataFrame with any fields overridden."""
    d = copy.deepcopy(_base)
    d.update(overrides)
    return pd.DataFrame(d)


def smoke(name: str, df_test: pd.DataFrame, should_pass: bool) -> None:
    """Run one validator smoke test and print PASS / FAIL."""
    try:
        v.validate(df_test)
        if should_pass:
            ok(f"{name}")
        else:
            fail(f"{name}  -- expected failure but passed!")
    except ValueError as e:
        if not should_pass:
            # grab the first bullet line from the multi-issue report
            msg = next(
                (ln.strip() for ln in str(e).splitlines() if ln.strip().startswith("•")),
                str(e)[:80],
            )
            ok(f"{name:<45}  caught: {msg[:65]}")
        else:
            fail(f"{name}  -- unexpected error: {e}")


smoke("Clean row passes validation",       make(),                                         True)
smoke("Empty dataframe raises",            pd.DataFrame(),                                 False)
smoke("Missing column raises",             make().drop(columns=["department"]),            False)
smoke("Invalid gender value",              make(gender=["Unknown"]),                       False)
smoke("Age out of range (200)",            make(age=[200]),                                False)
smoke("Charge amount = 0",                 make(charge_amount_USD=[0]),                    False)
smoke("Payment exceeds charge",            make(payment_amount_USD=[999]),                 False)
smoke("Bad appointment_id format",         make(appointment_id=["00001"]),                 False)
smoke("Inpatient missing admission_date",  make(visit_type=["Inpatient"],
                                               admission_date=[None],
                                               discharge_date=["10-01-2024"]),             False)
smoke("Discharge before admission",        make(visit_type=["Inpatient"],
                                               admission_date=["15-01-2024"],
                                               discharge_date=["10-01-2024"]),             False)
smoke("Inpatient with valid dates passes", make(visit_type=["Inpatient"],
                                               admission_date=["01-01-2024"],
                                               discharge_date=["10-01-2024"]),             True)

logger.enable("src.stage_1_mlops.data_ingestion.data_validator")


# ── Stage 8: data drift detection demo ───────────────────────────────────────

section("STAGE 8 - Data Drift Detection (JSD + PSI)")

from src.stage_1_mlops.data_drift import DataDriftDetector

detector = DataDriftDetector(reference_path="artifacts/reference_distribution.json")

# Simulate: first 80% = training / reference baseline
#           last 20%  = new production batch arriving later
split = int(len(df) * 0.80)
train_df = df.iloc[:split].copy()
prod_df  = df.iloc[split:].copy()

print(f"\n  Training slice  : {len(train_df)} rows  (reference)")
print(f"  Production slice: {len(prod_df)} rows  (current batch)")

# Step 1: fit on training data (saves reference_distribution.json)
detector.fit(train_df)
ok("Reference distribution saved to artifacts/reference_distribution.json")

# Step 2: detect drift on production slice
report = detector.detect(prod_df)
print()
print(report.summary())

if report.overall_drift:
    print("\n  [WARN]  Drift detected — review drifted columns before retraining.")
else:
    ok("No drift detected — production data is statistically consistent with training.")


# ── Stage 9: Data Transformation ─────────────────────────────────────────────

section("STAGE 9 - Data Transformation (Encoding)")

from src.stage_1_mlops.data_preprocessing import DataPreprocessingPipeline

preprocessing = DataPreprocessingPipeline(include_optional=True)
X = preprocessing.run_training(df)

ok(f"Output shape          : {X.shape[0]} rows x {X.shape[1]} columns")
ok(f"Columns               : {list(X.columns)}")

nulls = X.isnull().sum().sum()
if nulls == 0:
    ok("No nulls in transformed data")
else:
    fail(f"Nulls found in transformed data: {nulls}")

X_reloaded = DataPreprocessingPipeline.load_transformed_data()
if X_reloaded.shape == X.shape:
    ok("transformed_data.pkl saved and reloaded successfully")
else:
    fail("pkl reload shape mismatch")

print(f"\n  Sample (first 3 rows):\n{X.head(3).to_string()}")


# ── Stage 10: Model Training (3-track) ───────────────────────────────────────

section("STAGE 10 - Model Training  (Clustered IF + Global IF + Business Rules)")

from src.stage_1_mlops.model_training import ModelTrainer

trainer = ModelTrainer(n_estimators=100, contamination=0.01, random_state=42)
predictions = trainer.run()

ok(f"Predictions shape       : {predictions.shape}")
ok(f"Clusters found          : {predictions['cluster'].nunique()}")
ok(f"score_A range           : [{predictions['score_A'].min():.4f}, {predictions['score_A'].max():.4f}]")
ok(f"score_B range           : [{predictions['score_B'].min():.4f}, {predictions['score_B'].max():.4f}]")
ok(f"Business rule flags (C) : {int(predictions['flag_C'].sum())} records")
ok("kmeans.pkl + cluster_models.pkl + global_isolation_forest.pkl saved")
ok("predictions.pkl saved to artifacts/")


# ── Stage 11: Threshold Analysis ─────────────────────────────────────────────

section("STAGE 11 - Threshold Analysis  (99th percentile)")

from src.stage_1_mlops.model_evaluation import ThresholdAnalyzer

thresholds = ThresholdAnalyzer(percentile=99).run()

ok(f"Threshold Track A (clustered IF) : {thresholds['threshold_A']:.4f}")
ok(f"Threshold Track B (global IF)    : {thresholds['threshold_B']:.4f}")
ok("thresholds.json saved to artifacts/")


# ── Stage 12: Evaluation  (tier assignment + report) ─────────────────────────

section("STAGE 12 - Evaluation  (Tier Assignment + Score Distribution)")

from src.stage_1_mlops.model_evaluation import Evaluator, ScoreDistribution

tier_df = Evaluator().run()

tier_counts = tier_df["tier"].value_counts().sort_index()
ok(f"Tier 3 — investigate  : {tier_counts.get(3, 0):>5}  ({tier_counts.get(3, 0)/len(tier_df)*100:.2f}%)")
ok(f"Tier 2 — review       : {tier_counts.get(2, 0):>5}  ({tier_counts.get(2, 0)/len(tier_df)*100:.2f}%)")
ok(f"Tier 1 — monitor      : {tier_counts.get(1, 0):>5}  ({tier_counts.get(1, 0)/len(tier_df)*100:.2f}%)")
ok(f"Tier 0 — normal       : {tier_counts.get(0, 0):>5}  ({tier_counts.get(0, 0)/len(tier_df)*100:.2f}%)")
ok("tier_labels.pkl + evaluation_report.json saved to artifacts/")

ScoreDistribution().run()
ok("Score distribution logged")

tier3 = tier_df[tier_df["tier"] == 3]
if not tier3.empty:
    print(f"\n  Top 5 highest-confidence anomalies (Tier 3):")
    display_cols = ["cluster", "charge_amount_USD", "payment_amount_USD",
                    "charge_z_score", "score_A", "score_B", "tier"]
    print(tier3[display_cols]
          .sort_values("score_A", ascending=False)
          .head(5)
          .to_string())


# ── Stage 13: Inference (daily batch scoring) ─────────────────────────────────

section("STAGE 13 - Inference  (AnomalyScorer on simulated daily batch)")

from src.stage_1_mlops.inference import AnomalyScorer

# Use the last 20 % of the raw training data as a simulated new daily batch.
# In production this would be today's uploaded CSV; the API is identical.
batch_df = df.iloc[int(len(df) * 0.80):].copy()
print(f"\n  Simulated batch : {len(batch_df)} rows (last 20% of training data)")

scorer  = AnomalyScorer(include_optional=True)
results = scorer.run(batch_df, save_results=True)

tier_counts_inf = results["tier"].value_counts().sort_index()
ok(f"Inference output shape  : {results.shape}")
ok(f"Tier 3 — investigate    : {tier_counts_inf.get(3, 0):>4}")
ok(f"Tier 2 — review         : {tier_counts_inf.get(2, 0):>4}")
ok(f"Tier 1 — monitor        : {tier_counts_inf.get(1, 0):>4}")
ok(f"Tier 0 — normal         : {tier_counts_inf.get(0, 0):>4}")
ok("inference_results.pkl saved to artifacts/")
ok("drift_report.csv appended to artifacts/")


# ── Stage 14: Model Registry ──────────────────────────────────────────────────

section("STAGE 14 - Model Registry  (register trained model)")

import json as _json
from src.stage_1_mlops.model_registry import ModelRegistry

with open("artifacts/evaluation_report.json", encoding="utf-8") as _f:
    eval_report = _json.load(_f)

registry = ModelRegistry()
version  = registry.register(metrics=eval_report, notes="initial training run — 6000 records")

ok(f"Registered version      : {version}")
ok(f"Total versions in registry : {len(registry.list_versions())}")

latest = registry.get_latest()
ok(f"Latest tier-3 count     : {latest['metrics']['tier_distribution']['tier_3_investigate']['count']}")
ok("model_registry.json saved to artifacts/")


# ── Stage 15: AgentOps — Stage 2 daily report ────────────────────────────────

section("STAGE 15 - AgentOps  (Stage 2: daily anomaly report)")

from src.stage_2_agentops import AgentOpsRunner

runner_s2 = AgentOpsRunner()
s2_paths  = runner_s2.run(batch_df, source_name="dev_run_batch_20pct")

ok(f"Stage 2 JSON report : {s2_paths['json_path']}")
ok(f"Stage 2 markdown    : {s2_paths['md_path']}")

import json as _json2

with open(s2_paths["json_path"], encoding="utf-8") as _f2:
    s2_report = _json2.load(_f2)

s2_sum = s2_report["anomaly_summary"]
ok(f"Tier 3 critical     : {s2_sum['tier_3_investigate_immediately']}")
ok(f"Tier 2 review       : {s2_sum['tier_2_scheduled_review']}")
ok(f"Tier 1 monitor      : {s2_sum['tier_1_monitor']}")
ok(f"Overall anomaly rate: {s2_sum['overall_anomaly_rate_pct']}%")

if s2_report.get("tier_3_cases"):
    print(f"\n  Sample Tier 3 case (first one):")
    c = s2_report["tier_3_cases"][0]
    print(f"    patient_id      : {c.get('patient_id', 'N/A')}")
    print(f"    department      : {c.get('department', 'N/A')}")
    print(f"    charge_amount   : ${c.get('charge_amount_USD', 0):,.2f}")
    print(f"    payment_amount  : ${c.get('payment_amount_USD', 0):,.2f}")
    print(f"    rules_triggered : {c.get('rules_triggered', [])}")
    print(f"    score_A / score_B: {c.get('score_A')} / {c.get('score_B')}")


# ── Stage 16: Decision Making — Stage 3 stakeholder LLM ──────────────────────

section("STAGE 16 - Decision Making  (Stage 3: stakeholder LLM via Ollama / llama3.2)")

print(
    "\n  NOTE: Stage 3 requires Ollama running locally."
    "\n  If not installed: https://ollama.com  then  'ollama pull llama3.2'"
    "\n  If Ollama is offline, Stage 3 falls back gracefully with a manual-review directive.\n"
)

from src.stage_3_decision import DecisionRunner

runner_s3 = DecisionRunner()
s3_paths  = runner_s3.run(s2_paths["json_path"])

ok(f"Stage 3 JSON report : {s3_paths['json_path']}")
ok(f"Stage 3 markdown    : {s3_paths['md_path']}")

with open(s3_paths["json_path"], encoding="utf-8") as _f3:
    s3_report = _json2.load(_f3)

ds = s3_report.get("decision_summary", {})
ok(f"Risk level          : {ds.get('risk_level', 'N/A')}")
ok(f"Immediate actions   : {ds.get('immediate_action_count', 0)}")
ok(f"Dept directives     : {ds.get('department_directive_count', 0)}")
ok(f"Payer directives    : {ds.get('payer_directive_count', 0)}")

brief = s3_report.get("decision_brief", {})
exec_summary = brief.get("executive_summary", "")
if exec_summary:
    print(f"\n  Executive Summary (from LLM):")
    for line in exec_summary.split(". "):
        if line.strip():
            print(f"    {line.strip()}.")

actions = brief.get("immediate_actions", [])
if actions:
    print(f"\n  Immediate Actions ({len(actions)}):")
    for a in actions[:3]:
        print(f"    [{a.get('priority', '?')}] {a.get('action', '—')}")
        print(f"         Owner  : {a.get('owner', '—')}")
        print(f"         Reason : {a.get('reason', '—')[:90]}")


# ── Done ──────────────────────────────────────────────────────────────────────

section("ALL CHECKS COMPLETE")
print("  Stage 1  MLOps        — Ingestion + Validation + Drift + Transformation")
print("                          + 3-Track Model Training + Threshold + Evaluation")
print("                          + Inference (daily batch) + Model Registry")
print("  Stage 2  AgentOps     — Daily anomaly report (JSON + markdown)")
print("  Stage 3  Decision LLM — Stakeholder decision brief (JSON + markdown)\n")
