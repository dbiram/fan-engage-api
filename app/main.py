import os, io, uuid, tempfile, subprocess, datetime
from urllib.parse import urlparse
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from minio import Minio
from minio.error import S3Error

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://app:app@postgres:5432/app")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "0") == "1"
EXTERNAL_MINIO_ENDPOINT = os.getenv("EXTERNAL_MINIO_ENDPOINT")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
BUCKET_RAW = os.getenv("BUCKET_RAW", "raw")
BUCKET_FRAMES = os.getenv("BUCKET_FRAMES", "frames")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    object_key = Column(String, nullable=False)  # s3 key to the mp4
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

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

for b in (BUCKET_RAW, BUCKET_FRAMES):
    try:
        ensure_bucket(b)
    except S3Error:
        pass

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

    # Extract frames (1 fps for demo) to local temp dir, then upload to MinIO
    frames_dir = tempfile.mkdtemp()
    out_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    cmd = ["ffmpeg", "-y", "-i", tmp_path, "-vf", "fps=1", out_pattern]
    subprocess.run(cmd, check=True)

    # Upload frames
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.endswith(".jpg"): continue
        s3_internal.fput_object(BUCKET_FRAMES, f"match_{match.id}/{fname}", os.path.join(frames_dir, fname), content_type="image/jpeg")

    # Clean temp
    try: os.remove(tmp_path)
    except: pass

    return {"id": match.id, "title": match.title}
