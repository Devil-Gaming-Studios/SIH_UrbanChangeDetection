"""
Lightweight SQLite-backed history store for past analysis runs.
Keeps enough metadata + file paths to re-render a past result in the
frontend without re-running inference.
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "history.db"

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True)  # run_id (uuid hex)
    created_at = Column(DateTime, default=datetime.utcnow)

    year_earlier = Column(Integer, nullable=False)
    year_later = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)
    pixel_resolution_m = Column(Float, nullable=True)

    earlier_filename = Column(String, nullable=True)
    later_filename = Column(String, nullable=True)

    changed_percentage = Column(Float, nullable=False)
    annual_growth_rate_percentage = Column(Float, nullable=False)
    changed_area_km2 = Column(Float, nullable=True)

    # Full JSON blobs so the frontend can re-render exactly what /analyze returned.
    stats_json = Column(Text, nullable=False)
    images_json = Column(Text, nullable=False)
    growth_chart_url = Column(String, nullable=False)
    report_pdf_url = Column(String, nullable=False)
    thumbnail_url = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(engine)


def save_run(run_id, year_earlier, year_later, threshold, pixel_resolution_m,
             earlier_filename, later_filename, stats, images, growth_chart_url,
             report_pdf_url, thumbnail_url=None):
    session = SessionLocal()
    try:
        record = Run(
            id=run_id,
            year_earlier=year_earlier,
            year_later=year_later,
            threshold=threshold,
            pixel_resolution_m=pixel_resolution_m,
            earlier_filename=earlier_filename,
            later_filename=later_filename,
            changed_percentage=stats["changed_percentage"],
            annual_growth_rate_percentage=stats["annual_growth_rate_percentage"],
            changed_area_km2=stats.get("changed_area_km2"),
            stats_json=json.dumps(stats),
            images_json=json.dumps(images),
            growth_chart_url=growth_chart_url,
            report_pdf_url=report_pdf_url,
            thumbnail_url=thumbnail_url or images.get("overlay_on_newer"),
        )
        session.add(record)
        session.commit()
    finally:
        session.close()


def list_runs(limit=50):
    session = SessionLocal()
    try:
        rows = (
            session.query(Run)
            .order_by(Run.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "run_id": r.id,
                "created_at": r.created_at.isoformat() + "Z",
                "year_earlier": r.year_earlier,
                "year_later": r.year_later,
                "changed_percentage": r.changed_percentage,
                "annual_growth_rate_percentage": r.annual_growth_rate_percentage,
                "changed_area_km2": r.changed_area_km2,
                "thumbnail_url": r.thumbnail_url,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_run(run_id):
    session = SessionLocal()
    try:
        r = session.query(Run).filter(Run.id == run_id).first()
        if not r:
            return None
        return {
            "run_id": r.id,
            "created_at": r.created_at.isoformat() + "Z",
            "year_earlier": r.year_earlier,
            "year_later": r.year_later,
            "stats": json.loads(r.stats_json),
            "images": json.loads(r.images_json),
            "growth_chart": r.growth_chart_url,
            "report_pdf": r.report_pdf_url,
        }
    finally:
        session.close()


def delete_run(run_id):
    session = SessionLocal()
    try:
        r = session.query(Run).filter(Run.id == run_id).first()
        if r:
            session.delete(r)
            session.commit()
            return True
        return False
    finally:
        session.close()
