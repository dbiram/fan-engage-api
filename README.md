# Fan Engage — API (FastAPI)

FastAPI service for ingest, listing matches, and generating **presigned URLs** for video playback.
Also handles frame extraction via `ffmpeg` (Phase 1).

## Key idea: two MinIO clients
We use **two** MinIO connections:
- `s3_internal`: talks to MinIO **inside Docker network** (e.g., `minio:9000`) for uploads, listing, and frame writes.
- `s3_presign`: used **only** to generate presigned URLs with the **external endpoint** (e.g., `http://localhost:9000`).
  > The presigned URL includes host + region in the HMAC signature; they must match the URL the browser will call.

If host/region don’t match, the browser gets `SignatureDoesNotMatch`.

## Endpoints (Phase 1)
- `GET /health` – health probe
- `POST /ingest/video` – upload MP4, save DB row, extract frames (1fps)
- `GET /matches` – list matches with `video_url` (presigned)

## Local run (via compose)
This repo is intended to be run via `fan-engage-infra/docker-compose.dev.yml`.

## Dev tips
- If you change Python deps: rebuild only API  
  `docker compose -f ../fan-engage-infra/docker-compose.dev.yml build api && docker compose -f ../fan-engage-infra/docker-compose.dev.yml up -d api`
