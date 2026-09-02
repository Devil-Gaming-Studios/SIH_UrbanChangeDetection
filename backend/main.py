import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
from PIL import Image

from inference import (
    INFERENCE_VERSION,
    run_inference,
    run_msi_zip_inference,
    get_aligned_rgb_pair,
)
from report import (
    save_outputs,
    compute_growth_stats,
    make_growth_chart,
    build_pdf_report,
)
from image_metrics import compute_classical_metrics
import db


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "runs"
DATA_DIR.mkdir(exist_ok=True)

# Two independently trained checkpoints.
RGB_CHECKPOINT_PATH = BASE_DIR / "checkpoint_final.pth"
MSI_CHECKPOINT_PATH = BASE_DIR / "checkpoint_final (2).pth"


app = FastAPI(title="Urban Change Detection API")
db.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "inference_file": str((BASE_DIR / "inference.py").resolve()),
        "inference_version": INFERENCE_VERSION,
        "checkpoints": {
            "rgb": RGB_CHECKPOINT_PATH.exists(),
            "msi": MSI_CHECKPOINT_PATH.exists(),
        },
    }


@app.get("/models")
def models():
    return {
        "rgb": {
            "mode": "rgb",
            "input": "two JPG/PNG/RGB images",
            "checkpoint": RGB_CHECKPOINT_PATH.name,
            "available": RGB_CHECKPOINT_PATH.exists(),
        },
        "msi": {
            "mode": "msi",
            "input": "one ZIP containing earlier + later 13-band Sentinel-2 scenes",
            "checkpoint": MSI_CHECKPOINT_PATH.name,
            "available": MSI_CHECKPOINT_PATH.exists(),
        },
    }


def _select_checkpoint(mode: str) -> Path:
    mode = mode.lower().strip()

    if mode == "rgb":
        path = RGB_CHECKPOINT_PATH
    elif mode == "msi":
        path = MSI_CHECKPOINT_PATH
    else:
        raise HTTPException(
            400,
            "Invalid mode. Use 'rgb' or 'msi'.",
        )

    if not path.exists():
        raise HTTPException(
            500,
            f"Checkpoint for mode '{mode}' not found at {path}.",
        )

    return path


@app.post("/analyze")
async def analyze(
    mode: str = Form("rgb"),
    image_earlier: UploadFile | None = File(None),
    image_later: UploadFile | None = File(None),
    msi_zip: UploadFile | None = File(None),
    year_earlier: int = Form(...),
    year_later: int = Form(...),
    threshold: float = Form(0.5),
    pixel_resolution_m: float | None = Form(None),
):
    """
    Select the trained model through REST:

      mode=rgb
        image_earlier + image_later
        -> checkpoint_final.pth

      mode=msi
        msi_zip
        -> checkpoint_final (1).pth

    RGB mode remains the simple two-image workflow.
    MSI mode accepts one ZIP containing both dates.
    """
    mode = mode.lower().strip()

    if mode not in {"rgb", "msi"}:
        raise HTTPException(400, "mode must be 'rgb' or 'msi'.")

    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(400, "threshold must be between 0 and 1.")

    checkpoint_path = _select_checkpoint(mode)

    run_id = uuid.uuid4().hex[:10]
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    earlier_path = None
    later_path = None
    zip_path = None

    try:
        if mode == "rgb":
            if image_earlier is None or image_later is None:
                raise HTTPException(
                    400,
                    "RGB mode requires image_earlier and image_later.",
                )

            earlier_path = run_dir / (
                f"earlier_{Path(image_earlier.filename or 'earlier').name}"
            )
            later_path = run_dir / (
                f"later_{Path(image_later.filename or 'later').name}"
            )

            with open(earlier_path, "wb") as f:
                shutil.copyfileobj(image_earlier.file, f)

            with open(later_path, "wb") as f:
                shutil.copyfileobj(image_later.file, f)

            result = run_inference(
                checkpoint_path,
                earlier_path,
                later_path,
                mode="rgb",
                device=device,
                threshold=threshold,
            )

        else:
            if msi_zip is None:
                raise HTTPException(
                    400,
                    "MSI mode requires msi_zip.",
                )

            if not (msi_zip.filename or "").lower().endswith(".zip"):
                raise HTTPException(
                    400,
                    "MSI input must be a ZIP file.",
                )

            zip_path = run_dir / "msi_input.zip"

            with open(zip_path, "wb") as f:
                shutil.copyfileobj(msi_zip.file, f)

            result = run_msi_zip_inference(
                checkpoint_path,
                zip_path,
                device=device,
                threshold=threshold,
            )

            # The ZIP is intentionally retained in the run directory for
            # audit/reproducibility. No extracted band files are retained.

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            500,
            f"Inference failed in mode '{mode}': {exc}",
        )

    # The existing report layer expects earlier/later image paths.
    # RGB has the original uploaded files. MSI gets true-color previews.
    try:
        if mode == "rgb":
            image_paths = save_outputs(
                result,
                earlier_path,
                later_path,
                run_dir,
                stem="result",
                threshold=threshold,
            )

            earlier_rgb, later_rgb = get_aligned_rgb_pair(
                earlier_path,
                later_path,
                mode="rgb",
            )

        else:
            # Build temporary true-color previews from the MSI ZIP so the
            # existing report/output code can continue to operate.
            from inference import prepare_msi_zip

            earlier_stack, later_stack, tmp_dir = prepare_msi_zip(zip_path)

            try:
                earlier_preview = run_dir / "earlier_preview.png"
                later_preview = run_dir / "later_preview.png"

                # B04/B03/B02 -> RGB.
                earlier_rgb = earlier_stack[..., [3, 2, 1]]
                later_rgb = later_stack[..., [3, 2, 1]]

                Image.fromarray(
                    (earlier_rgb * 255).clip(0, 255).astype("uint8")
                ).save(earlier_preview)
                Image.fromarray(
                    (later_rgb * 255).clip(0, 255).astype("uint8")
                ).save(later_preview)

                image_paths = save_outputs(
                    result,
                    earlier_preview,
                    later_preview,
                    run_dir,
                    stem="result",
                    threshold=threshold,
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as exc:
        raise HTTPException(
            500,
            f"Output generation failed: {exc}",
        )

    stats = compute_growth_stats(
        result,
        year_earlier,
        year_later,
        pixel_resolution_m=pixel_resolution_m,
    )

    try:
        classical_stats, classical_images = compute_classical_metrics(
            earlier_rgb,
            later_rgb,
            run_dir,
            stem="result",
        )
        stats["classical_metrics"] = classical_stats
        image_paths.update(classical_images)
    except Exception:
        pass

    chart_path = run_dir / "growth_chart.png"
    make_growth_chart(stats, chart_path)

    pdf_path = run_dir / "report.pdf"
    build_pdf_report(
        pdf_path,
        image_paths,
        stats,
        chart_path,
        year_earlier,
        year_later,
    )

    def url_for(p: Path) -> str:
        return f"/files/{run_id}/{p.name}"

    images_urls = {
        k: url_for(v)
        for k, v in image_paths.items()
    }

    growth_chart_url = url_for(chart_path)
    report_pdf_url = url_for(pdf_path)

    db.save_run(
        run_id=run_id,
        year_earlier=year_earlier,
        year_later=year_later,
        threshold=threshold,
        pixel_resolution_m=pixel_resolution_m,
        earlier_filename=(
            image_earlier.filename
            if image_earlier is not None
            else (msi_zip.filename if msi_zip else "msi.zip")
        ),
        later_filename=(
            image_later.filename
            if image_later is not None
            else "msi-later-from-zip"
        ),
        stats=stats,
        images=images_urls,
        growth_chart_url=growth_chart_url,
        report_pdf_url=report_pdf_url,
    )

    return {
        "run_id": run_id,
        "mode": mode,
        "checkpoint": checkpoint_path.name,
        "stats": stats,
        "images": images_urls,
        "growth_chart": growth_chart_url,
        "report_pdf": report_pdf_url,
    }


@app.get("/report/{run_id}")
def get_report(run_id: str):
    pdf_path = DATA_DIR / run_id / "report.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Report not found")
    return {"report_pdf": f"/files/{run_id}/report.pdf"}


@app.get("/history")
def get_history(limit: int = 50):
    return {"runs": db.list_runs(limit=limit)}


@app.get("/history/{run_id}")
def get_history_run(run_id: str):
    record = db.get_run(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    return record


@app.delete("/history/{run_id}")
def delete_history_run(run_id: str):
    deleted = db.delete_run(run_id)
    if not deleted:
        raise HTTPException(404, "Run not found")

    run_dir = DATA_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    return {"deleted": True}
