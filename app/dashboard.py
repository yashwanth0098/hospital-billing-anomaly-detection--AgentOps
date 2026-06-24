"""
FastAPI dashboard — AgentOps daily anomaly report viewer.

Run:
    uvicorn app.dashboard:app --reload --port 8000
Then open: http://localhost:8000
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

_BASE_DIR       = Path(__file__).resolve().parent.parent
_REPORTS_DIR    = _BASE_DIR / "artifacts" / "agentops_reports"
_DECISIONS_DIR  = _BASE_DIR / "artifacts" / "decision_reports"
_DEFAULT_CSV    = _BASE_DIR / "src" / "stage_1_mlops" / "input_excel" / "Hospital_billing.csv"
_TEMPLATES_DIR  = Path(__file__).resolve().parent / "templates"

app       = FastAPI(title="AgentOps Anomaly Dashboard")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing page — lists past reports and offers a Run Pipeline button."""
    past_reports = _load_past_reports()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"past_reports": past_reports, "default_csv": _DEFAULT_CSV.name},
    )


@app.post("/run")
async def run_pipeline(file: UploadFile = File(None)):
    """
    Trigger the AgentOps pipeline.
    Uses the uploaded CSV if provided, otherwise the default billing CSV.
    Redirects to the report page on completion.
    """
    from src.stage_2_agentops import AgentOpsRunner
    runner = AgentOpsRunner()

    if file and file.filename:
        suffix  = Path(file.filename).suffix or ".csv"
        tmp     = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp.close()
            paths = runner.run(tmp.name, source_name=file.filename)
        finally:
            os.unlink(tmp.name)
    else:
        paths = runner.run(str(_DEFAULT_CSV))

    run_id = Path(paths["json_path"]).stem
    return RedirectResponse(f"/report/{run_id}", status_code=303)


@app.get("/report/{run_id}", response_class=HTMLResponse)
async def show_report(request: Request, run_id: str):
    """Render the full Stage 2 dashboard for a specific run."""
    json_path = _REPORTS_DIR / f"{run_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{run_id}' not found.")
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)
    decision_exists = (_DECISIONS_DIR / f"{run_id}_decisions.json").exists()
    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={"report": report, "decision_exists": decision_exists},
    )


@app.get("/decide/{run_id}")
async def trigger_decision(run_id: str):
    """
    Trigger Stage 3 Decision Agent on an existing Stage 2 report.
    Runs the full stakeholder LLM pipeline, then redirects to the decision page.
    """
    json_path = _REPORTS_DIR / f"{run_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Stage 2 report '{run_id}' not found.")

    from src.stage_3_decision import DecisionRunner
    runner = DecisionRunner()
    runner.run(str(json_path))

    return RedirectResponse(f"/decision/{run_id}", status_code=303)


@app.get("/decision/{run_id}", response_class=HTMLResponse)
async def show_decision(request: Request, run_id: str):
    """Render the Stage 3 stakeholder decision brief page."""
    json_path = _DECISIONS_DIR / f"{run_id}_decisions.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Decision report for '{run_id}' not found.")
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)
    return templates.TemplateResponse(
        request=request,
        name="decision.html",
        context={"report": report},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_past_reports() -> list[dict]:
    if not _REPORTS_DIR.exists():
        return []
    reports = []
    for path in sorted(_REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            s = r["anomaly_summary"]
            reports.append({
                "run_id":         r["run_id"],
                "run_date":       r["run_date"],
                "source":         r["batch_info"]["source"],
                "total_records":  r["batch_info"]["total_records"],
                "tier_3":         s["tier_3_investigate_immediately"],
                "tier_2":         s["tier_2_scheduled_review"],
                "tier_1":         s["tier_1_monitor"],
                "tier_0":         s["tier_0_normal"],
                "anomaly_rate":   s["overall_anomaly_rate_pct"],
            })
        except Exception:
            continue
    return reports


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app.dashboard:app", host="0.0.0.0", port=8000, reload=True)
