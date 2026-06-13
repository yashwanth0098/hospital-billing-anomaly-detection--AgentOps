EXPECTED_COLUMNS = [
    "patient_id", "gender", "age", "visit_date", "department",
    "physician_id", "diagnosis", "visit_type", "visit_reason",
    "appointment_id", "is_emergency", "insurance_id", "payer_name",
    "payer_type", "claim_status", "charge_amount_USD",
    "payment_amount_USD", "admission_date", "discharge_date",
]

DATE_COLUMNS = {
    "admission_date": "%d-%m-%Y",
    "discharge_date": "%d-%m-%Y",
    "visit_date":     "%d-%m-%Y",
}

MODE_IMPUTE_COLUMNS = [
    "diagnosis", "visit_reason", "payer_name",
    # admission_date / discharge_date are intentionally excluded:
    # Outpatient visits have no dates (NaT is correct); Inpatient missing
    # dates are flagged by the validator, not silently filled.
]

CONSTANT_IMPUTE_COLUMNS = {
    "insurance_id": "missing",
}

# ── Categorical allowed values ────────────────────────────────────────────────

ALLOWED_GENDER       = {"Male", "Female"}
ALLOWED_VISIT_TYPE   = {"Outpatient", "Inpatient"}
ALLOWED_IS_EMERGENCY = {"Yes", "No"}
ALLOWED_PAYER_TYPE   = {"Government", "Insurance", "Self-pay"}
ALLOWED_CLAIM_STATUS = {"Submitted", "Approved", "Denied", "Paid"}
ALLOWED_DEPARTMENTS  = {"Cardiology", "Orthopedics", "Pediatrics", "Neurology", "Oncology"}

# ── Numeric constraints ───────────────────────────────────────────────────────

NUMERIC_COLUMNS = ["patient_id", "physician_id", "age",
                   "charge_amount_USD", "payment_amount_USD"]

AGE_MIN = 0
AGE_MAX = 120

CHARGE_MIN = 0   # exclusive — charge must be > 0
PAYMENT_MIN = 0  # inclusive — payment can be 0 (not yet collected)

# ── Format patterns ───────────────────────────────────────────────────────────

APPOINTMENT_ID_PATTERN = r"^A\d{5}$"
