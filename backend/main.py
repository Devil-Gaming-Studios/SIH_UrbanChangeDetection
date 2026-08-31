import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch

from inference import run_inference
from report import save_outputs, compute_growth_stats, make_growth_chart, build_pdf_report

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "runs"
DATA_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = BASE_DIR / "checkpoint_final.pth"  # place your trained checkpoint here

app = FastAPI(title="Urban Change Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_found": CHECKPOINT_PATH.exists()}


@app.post("/analyze")
async def analyze(
    image_earlier: UploadFile = File(...),
    image_later: UploadFile = File(...),
    year_earlier: int = Form(...),
    year_later: int = Form(...),
    threshold: float = Form(0.5),
):
    if not CHECKPOINT_PATH.exists():
        raise HTTPException(500, f"Checkpoint not found at {CHECKPOINT_PATH}. Add your trained model file.")

    run_id = uuid.uuid4().hex[:10]
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    earlier_path = run_dir / f"earlier_{image_earlier.filename}"
    later_path = run_dir / f"later_{image_later.filename}"

    with open(earlier_path, "wb") as f:
        shutil.copyfileobj(image_earlier.file, f)
    with open(later_path, "wb") as f:
        shutil.copyfileobj(image_later.file, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        result = run_inference(
            CHECKPOINT_PATH, earlier_path, later_path, device=device, threshold=threshold
        )
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}")

    image_paths = save_outputs(result, earlier_path, later_path, run_dir, stem="result", threshold=threshold)
    stats = compute_growth_stats(result, year_earlier, year_later)
    chart_path = run_dir / "growth_chart.png"
    make_growth_chart(stats, chart_path)

    pdf_path = run_dir / "report.pdf"
    build_pdf_report(
        pdf_path, image_paths, stats, chart_path,
        year_earlier, year_later,
    )

    def url_for(p: Path) -> str:
        return f"/files/{run_id}/{p.name}"

    return {
        "run_id": run_id,
        "stats": stats,
        "images": {k: url_for(v) for k, v in image_paths.items()},
        "growth_chart": url_for(chart_path),
        "report_pdf": url_for(pdf_path),
    }


@app.get("/report/{run_id}")
def get_report(run_id: str):
    pdf_path = DATA_DIR / run_id / "report.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Report not found")
    return {"report_pdf": f"/files/{run_id}/report.pdf"}
