# fan-engage-api

FastAPI service for ingest, detections, team assignment, homography estimation, and analytics endpoints.

## Features
- `POST /ingest/video` → stores MP4 in MinIO, extracts frames
- `GET /analyze/detections` → runs detections (dev sync) — in prod use jobs
- `POST /teams/assign` → assigns teams per track
- `POST /homography/estimate` → estimates pitch homography per segment
- Analytics:
  - `GET /analytics/positions`
  - `GET /analytics/possession`
  - `GET /analytics/control_zones`
  - `GET /analytics/momentum`
- Jobs (via Redis/RQ):
  - `POST /jobs/pipeline?match_id=...`
  - `GET /jobs/{job_id}`
  - `POST /jobs/{job_id}/cancel`

## Architecture
- FastAPI + SQLAlchemy (Postgres)
- MinIO (S3-compatible) for raw videos, frames, detections, homography
- Ultralytics YOLO for detections
- Redis/RQ for background jobs (pipeline)

## Environment
Important vars (see `docker-compose.dev.yml` for full list):
```
DATABASE_URL=postgresql+psycopg2://app:app@postgres:5432/app
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123
MINIO_REGION=us-east-1
BUCKET_RAW=raw
BUCKET_FRAMES=frames
BUCKET_DETECTIONS=detections
BUCKET_TRACKS=tracks
BUCKET_HOMOGRAPHY=homography
BUCKET_ANALYTICS=analytics
BUCKET_MODELS=models
MODEL_LATEST_META=yolov8n_football/latest.json
BALL_MODEL_LATEST_META=ball_detection/latest.json
REDIS_URL=redis://redis:6379/0
```
## Run (dev via infra)
### from fan-engage-infra/
```
docker compose -f docker-compose.dev.yml up --build
```
### API at http://localhost:8000

## Endpoints (quick)
- `GET /matches`
- `POST /ingest/video`
- `GET /matches/{id}/detections`
- `POST /teams/assign?match_id=...`
- `POST /homography/estimate?match_id=...`
- Jobs: `POST /jobs/pipeline`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`

## Health
`GET /health` → `{ "status": "ok" }`
