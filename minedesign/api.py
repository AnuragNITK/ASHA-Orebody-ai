"""FastAPI service: four CSV paths (or uploads) in, one HTML report out.

Endpoints
---------
GET  /health            liveness plus which model paths are configured
GET  /config            the effective, non-secret configuration
POST /run               run the pipeline on four server-side file paths
POST /upload            run the pipeline on four uploaded CSVs
GET  /reports           list generated reports
GET  /reports/{name}    fetch one report as HTML
"""
from __future__ import annotations

import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import OUTPUT_DIR, SETTINGS
from .errors import MineDesignError
from .logging_utils import get_logger
from .pipeline import VERSION, PipelineConfig, run_pipeline

log = get_logger(__name__)

app = FastAPI(
    title="AI Mine Design System",
    version=VERSION,
    description=(
        "Replicates the Surpac open-pit workflow: collar, survey, assay and geology in; "
        "desurvey, compositing, domaining, block modelling and ultimate pit limit out, "
        "as one self-contained interactive HTML report. Exactly two LLM calls per run; "
        "every other step is deterministic."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("API_CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Economics(BaseModel):
    """Any value supplied here is treated as fact and never overridden by the model."""

    commodity_price: float | None = Field(None, gt=0)
    price_unit: str | None = None
    mining_cost_per_tonne: float | None = Field(None, gt=0)
    processing_cost_per_tonne: float | None = Field(None, gt=0)
    ga_cost_per_tonne: float | None = Field(None, ge=0)
    recovery_pct: float | None = Field(None, gt=0, le=100)
    overall_slope_angle_deg: float | None = Field(None, ge=10, le=85)
    discount_rate_pct: float | None = Field(None, ge=0, le=40)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class RunRequest(BaseModel):
    collar_path: str
    survey_path: str
    assay_path: str
    geology_path: str
    project_name: str = "Untitled Deposit"
    commodity: str = "gold"
    grade_unit: str = "g/t"
    density: float | None = Field(None, gt=0, le=10)
    composite_length: float | None = Field(None, gt=0)
    block_size: float | None = Field(None, gt=0)
    top_cut_percentile: float = Field(99.0, gt=50, le=100)
    run_cross_validation: bool = False
    economics: Economics = Field(default_factory=Economics)


class RunResponse(BaseModel):
    report_url: str
    report_file: str
    duration_s: float
    summary: dict
    calls: list
    caveat: str


CAVEAT = (
    "Decision support only. This is not a JORC or NI 43-101 mineral resource statement, "
    "which requires auditable methodology and Competent or Qualified Person sign-off. "
    "The cutoff grade, the variogram and estimator choice, and the pit economics are the "
    "three decisions where judgement and error live; have a mining engineer review them."
)


def _config_from(req: RunRequest, output_path: Path) -> PipelineConfig:
    return PipelineConfig(
        project_name=req.project_name,
        commodity=req.commodity,
        grade_unit=req.grade_unit,
        density=req.density,
        composite_length=req.composite_length,
        block_size=req.block_size,
        top_cut_percentile=req.top_cut_percentile,
        run_cross_validation=req.run_cross_validation,
        user_economics=req.economics.as_dict(),
        output_path=output_path,
    )


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_") or "report"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": VERSION,
        "paid_llm_configured": bool(SETTINGS.hf_token),
        "offline_mode": SETTINGS.offline,
        "models": {
            "call_1_orebody_grade": SETTINGS.model_orebody_grade,
            "call_2_pit_report": SETTINGS.model_pit_report,
            "local_fallback": SETTINGS.local_fallback_model,
        },
    }


@app.get("/config")
def get_config() -> dict:
    """Effective configuration, with the token redacted."""
    return {
        "hf_base_url": SETTINGS.hf_base_url,
        "hf_token_set": bool(SETTINGS.hf_token),
        "local_llm_base_url": SETTINGS.local_llm_base_url,
        "offline": SETTINGS.offline,
        "llm_timeout_s": SETTINGS.llm_timeout_s,
        "llm_max_retries": SETTINGS.llm_max_retries,
        "default_density": SETTINGS.default_density,
        "output_dir": str(OUTPUT_DIR),
    }


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    for label, path in (("collar", req.collar_path), ("survey", req.survey_path),
                        ("assay", req.assay_path), ("geology", req.geology_path)):
        if not Path(path).exists():
            raise HTTPException(400, f"{label} file not found: {path}")

    name = f"{_slug(req.project_name)}_{uuid.uuid4().hex[:8]}.html"
    output_path = OUTPUT_DIR / name

    try:
        result = run_pipeline(req.collar_path, req.survey_path, req.assay_path,
                              req.geology_path, config=_config_from(req, output_path))
    except MineDesignError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline crashed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"pipeline failed: {exc}") from exc

    return RunResponse(
        report_url=f"/reports/{name}",
        report_file=str(result.report_path),
        duration_s=round(result.duration_s, 2),
        summary=result.summary,
        calls=result.call_log,
        caveat=CAVEAT,
    )


@app.post("/upload", response_model=RunResponse)
async def upload(collar: UploadFile = File(...), survey: UploadFile = File(...),
                 assay: UploadFile = File(...), geology: UploadFile = File(...),
                 project_name: str = "Uploaded Deposit",
                 commodity: str = "gold") -> RunResponse:
    """Run the pipeline on four uploaded files (CSV or Excel .xlsx/.xlsm/.xls)."""
    workdir = Path(tempfile.mkdtemp(prefix="minedesign_"))
    try:
        paths = {}
        for label, upload_file in (("collar", collar), ("survey", survey),
                                   ("assay", assay), ("geology", geology)):
            if not upload_file.filename:
                raise HTTPException(400, f"{label} file is missing")
            # the loader dispatches on file extension, so the upload's own
            # suffix must survive -- forcing .csv would break an Excel upload
            suffix = Path(upload_file.filename).suffix.lower() or ".csv"
            target = workdir / f"{label}{suffix}"
            with target.open("wb") as fh:
                shutil.copyfileobj(upload_file.file, fh)
            paths[label] = str(target)

        request = RunRequest(collar_path=paths["collar"], survey_path=paths["survey"],
                             assay_path=paths["assay"], geology_path=paths["geology"],
                             project_name=project_name, commodity=commodity)
        return run(request)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/reports")
def list_reports() -> JSONResponse:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(OUTPUT_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return JSONResponse([
        {"name": p.name, "url": f"/reports/{p.name}",
         "size_mb": round(p.stat().st_size / 1e6, 2),
         "modified": int(p.stat().st_mtime)}
        for p in reports
    ])


@app.get("/reports/{name}")
def get_report(name: str) -> FileResponse:
    # resolve and confine to the output directory, so a crafted name cannot escape
    target = (OUTPUT_DIR / name).resolve()
    if OUTPUT_DIR.resolve() not in target.parents or not target.exists():
        raise HTTPException(404, "report not found")
    return FileResponse(target, media_type="text/html")


def main() -> None:
    import uvicorn

    uvicorn.run("minedesign.api:app",
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                reload=False)


if __name__ == "__main__":
    main()
