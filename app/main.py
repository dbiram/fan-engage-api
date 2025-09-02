import os, io, uuid, tempfile, subprocess, datetime, pathlib
from urllib.parse import urlparse
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .services.team_assignment import assign_teams_for_match

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel
import pandas as pd
import json
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
MODEL_LATEST_META = os.getenv("MODEL_LATEST_META", "yolov8n_football/latest.json")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/models/best.pt")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

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

def ensure_bucket(name: str):
    found = s3_internal.bucket_exists(name)
    if not found:
        s3_internal.make_bucket(name)

def _download_model_if_needed():
    # if local exists, keep it (you can add hash checks later)
    if os.path.exists(MODEL_LOCAL_PATH):
        return MODEL_LOCAL_PATH
    # fetch latest.json
    meta_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    try:
        s3_internal.fget_object(BUCKET_MODELS, MODEL_LATEST_META, meta_tmp)
    except Exception as e:
        print(f"[model] latest.json not found in MinIO ({BUCKET_MODELS}/{MODEL_LATEST_META}) -> using default yolov8n", e)
        return None
    with open(meta_tmp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_key = meta.get("best_pt")
    if not best_key:
        print("[model] best_pt missing in latest.json")
        return None
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    s3_internal.fget_object(BUCKET_MODELS, best_key, MODEL_LOCAL_PATH)
    print(f"[model] downloaded {best_key} -> {MODEL_LOCAL_PATH}")
    return MODEL_LOCAL_PATH

for b in (BUCKET_RAW, BUCKET_FRAMES, BUCKET_DETECTIONS, BUCKET_MODELS):
    try:
        ensure_bucket(b)
    except S3Error:
        pass

CUSTOM_MODEL_PATH = _download_model_if_needed()

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
    # For Phase 1, return a presigned URL for the MP4
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

    weights = CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH and os.path.exists(CUSTOM_MODEL_PATH) else "yolov8n.pt"
    model = YOLO(weights)
    print("Model class names:", model.names)
    print(f"[Detection] Running on {len(local_frames)} frames with conf_thres={conf_thres}, weights={weights}")

    results_gen = model.track(
        source=tmp_dir,
        stream=True,
        conf=conf_thres,
        imgsz=1280,
        tracker="bytetrack.yaml",
        persist=True,   # keep IDs across frames
        verbose=False,
    )

    rows = []
    frame_index = {os.path.basename(p): idx for idx, p in enumerate(local_frames)}
    total_frames = len(local_frames)
    checkpoint_interval = max(1, total_frames // 10)
    processed_frames = 0

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
            print(f"[Detection] Processed {processed_frames}/{total_frames} frames ({(processed_frames/total_frames)*100:.1f}%)")

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