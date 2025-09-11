import os, io, uuid, tempfile, subprocess, datetime, pathlib
from urllib.parse import urlparse
from typing import List
import math
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .services.team_assignment import assign_teams_for_match
from .services.homography import estimate_homography_for_match
from .services.analytics import (
    compute_positions_df,
    possession_timeseries,
    control_area_timeseries,
    momentum_timeseries,
    _try_load_analytics,
    _save_analytics,
    FPS
)

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel
import pandas as pd
import json

from redis import Redis
from rq import Queue, Retry
from rq.job import Job

from torch.serialization import add_safe_globals
try:
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel
    add_safe_globals([DetectionModel, SegmentationModel, PoseModel])
except Exception:
    try:
        from ultralytics.nn.tasks import DetectionModel
        add_safe_globals([DetectionModel])
    except Exception:
        pass

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://app:app@postgres:5432/app")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "0") == "1"
EXTERNAL_MINIO_ENDPOINT = os.getenv("EXTERNAL_MINIO_ENDPOINT")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
BUCKET_RAW = os.getenv("BUCKET_RAW", "raw")
BUCKET_FRAMES = os.getenv("BUCKET_FRAMES", "frames")
BUCKET_DETECTIONS = os.getenv("BUCKET_DETECTIONS", "detections")
BUCKET_TRACKS = os.getenv("BUCKET_TRACKS", "tracks")
BUCKET_MODELS = os.getenv("BUCKET_MODELS", "models")
BUCKET_HOMOGRAPHY = os.getenv("BUCKET_HOMOGRAPHY", "homography")
BUCKET_ANALYTICS = os.getenv("BUCKET_ANALYTICS", "analytics")
MODEL_LATEST_META = os.getenv("MODEL_LATEST_META", "yolov8n_football/latest.json")
BALL_MODEL_LATEST_META = os.getenv("BALL_MODEL_LATEST_META", "ball_detection/latest.json")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/models/best.pt")
BALL_MODEL_LOCAL_PATH = os.getenv("BALL_MODEL_LOCAL_PATH", "/models/ball_detection/best.pt")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
JOBS_QUEUE = os.getenv("JOBS_QUEUE", "default")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    object_key = Column(String, nullable=False)  # s3 key to the mp4
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DetectionRecord(BaseModel):
    frame_id: int
    filename: str
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    object_id: int | None

Base.metadata.create_all(engine)

def make_minio_client(endpoint: str, access_key: str, secret_key: str, default_secure: bool, region: str | None = None):
    # Accepte "host:port" OU une URL "http(s)://host:port"
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        u = urlparse(endpoint)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        secure = (u.scheme == "https")
        ep = f"{host}:{port}"
    else:
        ep = endpoint
        secure = default_secure
    return Minio(ep, access_key=access_key, secret_key=secret_key, secure=secure, region=region)

s3_internal = make_minio_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE, region=MINIO_REGION)

s3_presign = make_minio_client(
    EXTERNAL_MINIO_ENDPOINT if EXTERNAL_MINIO_ENDPOINT else MINIO_ENDPOINT,
    MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_SECURE, 
    region=MINIO_REGION
)

redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(JOBS_QUEUE, connection=redis_conn)

def _job_or_404(job_id: str) -> Job:
    try:
        return Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="job not found")

def ensure_bucket(name: str):
    found = s3_internal.bucket_exists(name)
    if not found:
        s3_internal.make_bucket(name)

def _download_model_if_needed(model_meta: str, local_path: str):
    # if local exists, keep it (you can add hash checks later)
    if os.path.exists(local_path):
        return local_path
    # fetch latest.json
    meta_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    try:
        s3_internal.fget_object(BUCKET_MODELS, model_meta, meta_tmp)
    except Exception as e:
        print(f"[model] latest.json not found in MinIO ({BUCKET_MODELS}/{model_meta}) -> using default yolov8n", e)
        return None
    with open(meta_tmp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_key = meta.get("best_pt")
    if not best_key:
        print("[model] best_pt missing in latest.json")
        return None
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_internal.fget_object(BUCKET_MODELS, best_key, local_path)
    print(f"[model] downloaded {best_key} -> {local_path}")
    return local_path

for b in (BUCKET_RAW, BUCKET_FRAMES, BUCKET_DETECTIONS, BUCKET_MODELS, BUCKET_HOMOGRAPHY, BUCKET_TRACKS, BUCKET_ANALYTICS):
    try:
        ensure_bucket(b)
    except S3Error:
        pass

CUSTOM_MODEL_PATH = _download_model_if_needed(MODEL_LATEST_META, MODEL_LOCAL_PATH)
BALL_MODEL_PATH = _download_model_if_needed(BALL_MODEL_LATEST_META, BALL_MODEL_LOCAL_PATH)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/matches")
def list_matches():
    print("EXTERNAL_MINIO_ENDPOINT:", EXTERNAL_MINIO_ENDPOINT)
    with SessionLocal() as db:
        rows = db.query(Match).order_by(Match.created_at.desc()).all()
    out = []
    for m in rows:
        url = s3_presign.presigned_get_object(BUCKET_RAW, m.object_key, expires=datetime.timedelta(hours=1))
        out.append({"id": m.id, "title": m.title, "video_url": url})
    return out

@app.post("/ingest/video")
def ingest_video(title: str = Form(...), file: UploadFile = File(...)):
    # Save to temp, upload to MinIO, record in DB, extract frames
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    key = f"matches/{uuid.uuid4().hex}.mp4"
    # upload original MP4
    s3_internal.fput_object(BUCKET_RAW, key, tmp_path, content_type="video/mp4")

    with SessionLocal() as db:
        match = Match(title=title, object_key=key)
        db.add(match); db.commit(); db.refresh(match)

    # Extract frames (25 fps for demo) to local temp dir, then upload to MinIO
    frames_dir = tempfile.mkdtemp()
    out_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    cmd = ["ffmpeg", "-y", "-i", tmp_path, "-vf", "fps=25", out_pattern]
    subprocess.run(cmd, check=True)

    # Upload frames
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.endswith(".jpg"): continue
        s3_internal.fput_object(BUCKET_FRAMES, f"match_{match.id}/{fname}", os.path.join(frames_dir, fname), content_type="image/jpeg")

    # Clean temp
    try: os.remove(tmp_path)
    except: pass

    return {"id": match.id, "title": match.title}

def run_detection(match_id: int, conf_thres: float = 0.25) -> str:
    """
    Run detection on all frames for a match and upload results to MinIO.
    Returns the MinIO key for the Parquet file.
    """
    if YOLO is None:
        # ultralytics not installed in API image
        raise HTTPException(status_code=500, detail="ultralytics not installed")

    prefix = f"match_{match_id}/"
    objects = s3_internal.list_objects(BUCKET_FRAMES, prefix=prefix, recursive=True)
    frame_keys = sorted([o.object_name for o in objects if o.object_name.endswith((".jpg", ".png"))])

    if not frame_keys:
        raise HTTPException(status_code=404, detail="No frames found for this match")

    tmp_dir = tempfile.mkdtemp(prefix=f"detect_{match_id}_")
    local_frames = []

    for key in frame_keys:
        local_img = os.path.join(tmp_dir, os.path.basename(key))
        s3_internal.fget_object(BUCKET_FRAMES, key, local_img)
        local_frames.append(local_img)

    # Load both models
    weights = CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH and os.path.exists(CUSTOM_MODEL_PATH) else "yolov8n.pt"
    ball_weights = BALL_MODEL_PATH if BALL_MODEL_PATH and os.path.exists(BALL_MODEL_PATH) else None
    
    model = YOLO(weights)
    ball_model = YOLO(ball_weights) if ball_weights else None
    
    print("Main model class names:", model.names)
    if ball_model:
        print("Ball model class names:", ball_model.names)
    print(f"[Detection] Running on {len(local_frames)} frames with conf_thres={conf_thres}, weights={weights}")

    rows = []
    frame_index = {os.path.basename(p): idx for idx, p in enumerate(local_frames)}
    total_frames = len(local_frames)
    checkpoint_interval = max(1, total_frames // 10)
    processed_frames = 0

    # Run main model first
    results_gen = model.track(
        source=tmp_dir,
        stream=True,
        conf=conf_thres,
        imgsz=1280,
        tracker="bytetrack.yaml",
        persist=True,   # keep IDs across frames
        verbose=False,
    )

    for r in results_gen:
        basename = os.path.basename(getattr(r, "path", ""))  # e.g., frame_00010.jpg
        frame_id = frame_index.get(basename, None)

        if frame_id is None or r.boxes is None:
            continue

        boxes = r.boxes
        # boxes.id may be None if tracker didn't assign an ID (e.g., first frames)
        ids = boxes.id.cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(boxes)
        clss = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else [None] * len(boxes)
        confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [None] * len(boxes)
        xyxys = boxes.xyxy.cpu().tolist()

        for obj_id, cls_idx, conf, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxys):
            class_name = model.names[int(cls_idx)] if cls_idx is not None else "object"
            if class_name != "ball":  # Skip balls from main model
                rows.append({
                    "frame_id": int(frame_id),
                    "filename": f"{prefix}{basename}",
                    "class_name": class_name,
                    "conf": float(conf) if conf is not None else None,
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "object_id": int(obj_id) if obj_id is not None else None,
                })
        processed_frames += 1
        if processed_frames % checkpoint_interval == 0 or processed_frames == total_frames:
            print(f"[Players Detection] Processed {processed_frames}/{total_frames} frames ({(processed_frames/total_frames)*100:.1f}%)")

    # Run ball detection model if available (without tracking)
    if ball_model:
        ball_results_gen = ball_model.predict(
            source=tmp_dir,
            stream=True,
            conf=conf_thres,
            imgsz=1280,
            verbose=False,
        )
        processed_frames = 0

        for r in ball_results_gen:
            basename = os.path.basename(getattr(r, "path", ""))
            frame_id = frame_index.get(basename, None)

            if frame_id is None or r.boxes is None:
                continue

            boxes = r.boxes
            clss = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else [None] * len(boxes)
            confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [None] * len(boxes)
            xyxys = boxes.xyxy.cpu().tolist()

            # Take the highest confidence ball detection in each frame
            if len(confs) > 0:
                best_idx = max(range(len(confs)), key=lambda i: confs[i])
                cls_idx = clss[best_idx]
                conf = confs[best_idx]
                x1, y1, x2, y2 = xyxys[best_idx]

                rows.append({
                    "frame_id": int(frame_id),
                    "filename": f"{prefix}{basename}",
                    "class_name": "ball",  # Force class name to ball
                    "conf": float(conf) if conf is not None else None,
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "object_id": 99999,  # Hardcoded ID for ball
                })
            processed_frames += 1
            if processed_frames % checkpoint_interval == 0 or processed_frames == total_frames:
                print(f"[Ball Detection] Processed {processed_frames}/{total_frames} frames ({(processed_frames/total_frames)*100:.1f}%)")

        

    df = pd.DataFrame(rows)
    if df.empty:
        raise HTTPException(status_code=422, detail="No detections produced")
    out_filename = f"match_{match_id}.parquet"
    local_out = os.path.join(tmp_dir, out_filename)
    df.to_parquet(local_out)

    dest_key = f"match_{match_id}/{out_filename}"
    s3_internal.fput_object(BUCKET_DETECTIONS, dest_key, local_out)
    return dest_key

@app.get("/analyze/detections")
def analyze_detections(match_id: int):
    """
    Synchronously run detection for a match and return the MinIO key to the Parquet file.
    (We’ll move this to a background worker in a later phase.)
    """
    key = run_detection(match_id, conf_thres=0.1)
    return {"match_id": match_id, "key": key}

@app.get("/matches/{match_id}/detections", response_model=List[DetectionRecord])
def get_detections(match_id: int):
    """
    Download detections Parquet for this match from MinIO and return as JSON.
    """
    prefix = f"match_{match_id}/"
    objects = s3_internal.list_objects(BUCKET_DETECTIONS, prefix=prefix, recursive=True)
    file_key = None
    for obj in objects:
        if obj.object_name.endswith(".parquet"):
            file_key = obj.object_name
            break
    if not file_key:
        raise HTTPException(status_code=404, detail="Detections not found. Run /analyze/detections first.")

    import tempfile, os
    tmp_dir = tempfile.mkdtemp(prefix=f"detections_{match_id}_")
    local_file = os.path.join(tmp_dir, os.path.basename(file_key))
    s3_internal.fget_object(BUCKET_DETECTIONS, file_key, local_file)

    df = pd.read_parquet(local_file)

    # Ensure object_id is int or None
    if "object_id" in df.columns:
        # First coerce to pandas nullable Int64, then convert NA to None
        df["object_id"] = df["object_id"].astype("Int64").astype(object).where(df["object_id"].notna(), None)

    # Replace any remaining NaN in the dataframe with None (safe for JSON)
    df = df.where(pd.notnull(df), None)

    # Convert to list of dicts
    records = df.to_dict(orient="records")
    return records

@app.post("/teams/assign")
def teams_assign(match_id: int):
    try:
        result = assign_teams_for_match(match_id, s3_internal)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/matches/{match_id}/tracks")
def get_tracks(match_id: int):
    key = f"match_{match_id}/match_{match_id}.parquet"
    try:
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, pathlib.Path(key).name)
            s3_internal.fget_object(BUCKET_TRACKS, key, dst)
            df = pd.read_parquet(dst, engine="pyarrow")
    except Exception:
        # Not found (or bucket missing)
        raise HTTPException(status_code=404, detail="tracks not found")
    return JSONResponse(df.to_dict(orient="records"))

@app.post("/homography/estimate")
def homography_estimate(match_id: int, segment_frames: int = 250, step: int = 5, conf_th: float = 0.5):
    try:
        result = estimate_homography_for_match(match_id, segment_frames, step, conf_th, s3_internal)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matches/{match_id}/homography")
def homography_list(match_id: int):
    prefix = f"match_{match_id}/"
    objs = list(s3_internal.list_objects(BUCKET_HOMOGRAPHY, prefix=prefix, recursive=True))
    if not objs:
        raise HTTPException(status_code=404, detail="no homography found")
    items = []
    with tempfile.TemporaryDirectory() as td:
        for o in objs:
            dst = os.path.join(td, pathlib.Path(o.object_name).name)
            s3_internal.fget_object(BUCKET_HOMOGRAPHY, o.object_name, dst)
            with open(dst, "r") as f:
                items.append(json.load(f))
    # sort by segment_index
    items.sort(key=lambda x: x.get("segment_index", 0))
    return JSONResponse(items)

@app.post("/analytics/positions")
def analytics_positions(match_id: int, bottom_center: bool = True):
    """
    Compute and save positions to analytics bucket.
    Returns success message and analytics key.
    """
    try:
        df = compute_positions_df(
            s3_internal, match_id,
            BUCKET_DETECTIONS, BUCKET_TRACKS, BUCKET_HOMOGRAPHY, BUCKET_ANALYTICS,
            bottom_center=bottom_center
        )
        return JSONResponse({
            "match_id": match_id,
            "analysis_type": "positions",
            "rows_computed": len(df),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/{match_id}/positions")
def get_analytics_positions(match_id: int, limit: int | None = None):
    """
    Get cached positions from analytics bucket.
    Returns rows: frame_id, object_id, class_name, team_id, x_norm,y_norm,x_m,y_m
    """
    try:
        df = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "positions")
        if df is None:
            raise HTTPException(status_code=404, detail="Positions not found. Run POST /analytics/positions first.")
        
        cols = ["frame_id","object_id","class_name","team_id","x_norm","y_norm","x_m","y_m"]
        df = df[cols].dropna(subset=["x_norm","y_norm"])
        if limit is not None:
            df = df.head(int(limit))
        
        # Replace any remaining NaN values with None for JSON compatibility
        df = df.replace({np.nan: None})
        
        return JSONResponse(df.to_dict(orient="records"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analytics/possession")
def analytics_possession(
    match_id: int,
    max_dist_m: float = 4.0,
    delta_margin_m: float = 0.5,
):
    """
    Compute and save nearest-ball possession with hysteresis to analytics bucket.
    Returns success message and analytics key.
    """
    try:
        # Try to load cached possession data first
        cached = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "possession")
        if cached is not None:
            return JSONResponse({
                "match_id": match_id,
                "analysis_type": "possession",
                "status": "already_exists",
                "rows_computed": len(cached)
            })
        
        # Try to load cached positions first, if not compute them
        pos = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "positions")
        if pos is None:
            pos = compute_positions_df(s3_internal, match_id, BUCKET_DETECTIONS, BUCKET_TRACKS, BUCKET_HOMOGRAPHY, BUCKET_ANALYTICS, bottom_center=True)
        
        ts, summary = possession_timeseries(pos, max_dist_m=max_dist_m, delta_margin_m=delta_margin_m, fps=FPS)
        # Save possession results
        _save_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "possession", ts)
        
        return JSONResponse({
            "match_id": match_id,
            "analysis_type": "possession",
            "rows_computed": len(ts),
            "summary": summary,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/{match_id}/possession")
def get_analytics_possession(match_id: int):
    """
    Get cached possession analysis from analytics bucket.
    Returns timeseries and summary.
    """
    try:
        ts = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "possession")
        if ts is None:
            raise HTTPException(status_code=404, detail="Possession analysis not found. Run POST /analytics/possession first.")
        
        # Recalculate summary from cached data
        counts = ts["team"].value_counts(dropna=True).to_dict()
        total = int(ts["team"].notna().sum())
        pct = {int(k): round(v*100.0/total, 2) for k,v in counts.items()} if total else {}
        summary = {"frames_counted": total, "percent_by_team": pct}
        
        # Ensure all values are JSON-serializable
        ts = ts.where(pd.notnull(ts), None)
        records = ts.to_dict(orient="records")
        # Clean any remaining non-finite values
        for record in records:
            for key, value in record.items():
                if isinstance(value, float) and (pd.isna(value) or not math.isfinite(value)):
                    record[key] = None
        return JSONResponse({"series": records, "summary": summary})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analytics/control_zones")
def analytics_control_zones(
    match_id: int,
    stride: int = FPS,   # compute ~each second
):
    """
    Compute and save Voronoi territory share (area %) per team over time to analytics bucket.
    Returns success message and analytics key.
    """
    try:
        # Try to load cached control zones data first
        cached = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "control_zones")
        if cached is not None:
            return JSONResponse({
                "match_id": match_id,
                "analysis_type": "control_zones",
                "status": "already_exists",
                "rows_computed": len(cached)
            })
        
        # Try to load cached positions first, if not compute them
        pos = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "positions")
        if pos is None:
            pos = compute_positions_df(s3_internal, match_id, BUCKET_DETECTIONS, BUCKET_TRACKS, BUCKET_HOMOGRAPHY, BUCKET_ANALYTICS, bottom_center=True)
        
        ctrl = control_area_timeseries(pos, stride=stride)
        # Save control zones results
        _save_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "control_zones", ctrl)
        
        return JSONResponse({
            "match_id": match_id,
            "analysis_type": "control_zones",
            "rows_computed": len(ctrl),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/{match_id}/control_zones")
def get_analytics_control_zones(match_id: int):
    """
    Get cached control zones analysis from analytics bucket.
    Returns Voronoi territory share (area %) per team over time.
    """
    try:
        ctrl = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "control_zones")
        if ctrl is None:
            raise HTTPException(status_code=404, detail="Control zones analysis not found. Run POST /analytics/control_zones first.")
        
        # Replace any NaN values with None for JSON compatibility
        ctrl = ctrl.replace({np.nan: None})
        
        return JSONResponse({"series": ctrl.to_dict(orient="records")})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analytics/momentum")
def analytics_momentum(
    match_id: int,
    stride: int = FPS,
    alpha: float = 0.1,
):
    """
    Compute and save momentum index from territory (EWMA-smoothed area share) to analytics bucket.
    Returns success message and analytics key.
    """
    try:
        # Try to load cached momentum data first
        cached = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "momentum")
        if cached is not None:
            return JSONResponse({
                "match_id": match_id,
                "analysis_type": "momentum",
                "status": "already_exists",
                "rows_computed": len(cached)
            })
        
        # Try to load cached control zones first
        ctrl = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "control_zones")
        if ctrl is None:
            # Try to load cached positions first, if not compute them
            pos = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "positions")
            if pos is None:
                pos = compute_positions_df(s3_internal, match_id, BUCKET_DETECTIONS, BUCKET_TRACKS, BUCKET_HOMOGRAPHY, BUCKET_ANALYTICS, bottom_center=True)
            
            ctrl = control_area_timeseries(pos, stride=stride)
            # Save control zones results
            _save_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "control_zones", ctrl)
        
        mom = momentum_timeseries(ctrl, alpha=alpha)
        # Save momentum results
        _save_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "momentum", mom)
        
        return JSONResponse({
            "match_id": match_id,
            "analysis_type": "momentum",
            "rows_computed": len(mom),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/{match_id}/momentum")
def get_analytics_momentum(match_id: int):
    """
    Get cached momentum analysis from analytics bucket.
    Returns momentum index from territory (EWMA-smoothed area share).
    """
    try:
        mom = _try_load_analytics(s3_internal, BUCKET_ANALYTICS, match_id, "momentum")
        if mom is None:
            raise HTTPException(status_code=404, detail="Momentum analysis not found. Run POST /analytics/momentum first.")
        
        # Replace any NaN values with None for JSON compatibility
        mom = mom.replace({np.nan: None})
        
        return JSONResponse({"series": mom.to_dict(orient="records")})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/jobs/pipeline")
def create_pipeline_job(match_id: int, conf_thres: float = 0.1):
    # optional idempotency: avoid duplicates
    job = queue.enqueue(
        "fe_workers.tasks.pipeline.job_pipeline",
        match_id,
        conf_thres=conf_thres,
        retry=Retry(max=int(os.getenv("JOBS_MAX_RETRIES", "2"))),
        job_timeout=int(os.getenv("JOBS_DEFAULT_TIMEOUT", "7200")),  # valid here
        description=f"pipeline:{match_id}",
    )
    return {"job_id": job.id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _job_or_404(job_id)
    meta = job.meta or {}
    return {
        "id": job.id,
        "status": job.get_status(),
        "progress": meta.get("progress", 0),
        "note": meta.get("note", ""),
        "result": job.result if job.is_finished else None,
        "exc": job.exc_info if job.is_failed else None,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
    }

@app.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    job = _job_or_404(job_id)
    job.cancel()
    return {"ok": True}