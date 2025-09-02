import os, tempfile, pathlib
from typing import List, Dict
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from minio import Minio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import open_clip

# --- Config from env (same names you used in ML prototype) ---
MATCH_PARAM_NAME = "match_id"  # endpoint query param name

BUCKET_DETECTIONS = os.getenv("BUCKET_DETECTIONS", "detections")
BUCKET_FRAMES     = os.getenv("BUCKET_FRAMES",     "frames")
BUCKET_TRACKS     = os.getenv("BUCKET_TRACKS",     "tracks")

MAX_SAMPLES_PER_TRACK = int(os.getenv("MAX_SAMPLES_PER_TRACK", "8"))
USE_UMAP = os.getenv("USE_UMAP", "1") == "1"
UMAP_COMPONENTS = int(os.getenv("UMAP_COMPONENTS", "16"))
ALPHA_COLOR = float(os.getenv("ALPHA_COLOR", "0.3"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Parquet via temp files (consistency with your style) ---
def _read_parquet(bucket: str, key: str, cli: Minio) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, pathlib.Path(key).name)
        cli.fget_object(bucket, key, dst)
        return pd.read_parquet(dst, engine="pyarrow")

def _write_parquet(bucket: str, key: str, df: pd.DataFrame, cli: Minio):
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, pathlib.Path(key).name)
        df.to_parquet(dst, engine="pyarrow", index=False)
        cli.fput_object(bucket, key, dst, content_type="application/octet-stream")

def _read_frame(filename: str, cli: Minio) -> np.ndarray:
    obj = cli.get_object(BUCKET_FRAMES, filename)  # filename contains full key (e.g., "match_14/frame_000123.jpg")
    arr = np.frombuffer(obj.read(), dtype=np.uint8)
    obj.close(); obj.release_conn()
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _crop_jersey(img_bgr: np.ndarray, x1, y1, x2, y2) -> Image.Image:
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w, int(x2)); y2 = min(h, int(y2))
    top = y1
    bot = y1 + max(1, int(0.6 * (y2 - y1)))
    crop = img_bgr[top:bot, x1:x2]
    if crop.size == 0:
        crop = img_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _hsv_hist(img_bgr: np.ndarray, x1, y1, x2, y2, bins=(12,6,6)) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w, int(x2)); y2 = min(h, int(y2))
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((bins[0]*bins[1]*bins[2],), dtype=np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist

def _load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEVICE)
    model.eval()
    return model, preprocess

def _embed_batch(model, preprocess, pil_images: List[Image.Image]) -> np.ndarray:
    with torch.no_grad():
        batch = torch.stack([preprocess(im) for im in pil_images]).to(DEVICE)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()

def assign_teams_for_match(match_id: int, cli: Minio) -> Dict:
    # 1) detections parquet path — same pattern as your ML code
    det_key = f"match_{match_id}/match_{match_id}.parquet"
    df = _read_parquet(BUCKET_DETECTIONS, det_key, cli)
    df = df[(df["class_name"] == "player") & df["object_id"].notna()].copy()
    if df.empty:
        return {"match_id": match_id, "tracks": 0, "by_team": {}, "note": "no player detections with object_id"}
    df["object_id"] = df["object_id"].astype(int)

    # 2) sampling + crops
    model, preprocess = _load_clip()
    samples = []  # (oid, frame_id, pil_crop, hist)
    total_objects = len(df["object_id"].unique())
    for idx, (oid, grp) in enumerate(df.groupby("object_id")):
        # Print progress every 10%
        progress = (idx + 1) / total_objects * 100
        if progress % 10 == 0:
            print(f"Processing tracks: {int(progress)}% complete")
        g = grp.sort_values("frame_id")
        step = max(1, len(g)//MAX_SAMPLES_PER_TRACK)
        g = g.iloc[::step][:MAX_SAMPLES_PER_TRACK]
        for _, r in g.iterrows():
            frame = _read_frame(r["filename"], cli)  # filename contains the full frames key
            pil = _crop_jersey(frame, r["x1"], r["y1"], r["x2"], r["y2"])
            hist = _hsv_hist(frame, r["x1"], r["y1"], r["x2"], r["y2"])
            samples.append((oid, r["frame_id"], pil, hist))
    if not samples:
        return {"match_id": match_id, "tracks": 0, "by_team": {}, "note": "no samples collected"}

    # 3) embeddings (batched)
    pil_list = [s[2] for s in samples]
    feats_e = []
    B = 32
    for i in range(0, len(pil_list), B):
        feats_e.append(_embed_batch(model, preprocess, pil_list[i:i+B]))
    feats_e = np.concatenate(feats_e, axis=0)

    # 4) aggregate per track
    rows = []
    oids = sorted(set(s[0] for s in samples))
    for oid in oids:
        idx = [i for i,(o,_,_,_) in enumerate(samples) if o == oid]
        e = feats_e[idx].mean(axis=0)
        h = np.stack([samples[i][3] for i in idx]).mean(axis=0)
        h = h / (np.linalg.norm(h) + 1e-8)
        feat = np.concatenate([e, ALPHA_COLOR * h], axis=0)
        rows.append((oid, len(idx), feat))
    X = np.stack([r[2] for r in rows], axis=0)
    ns = [r[1] for r in rows]

    # 5) reduce (optional UMAP) else PCA
    if USE_UMAP:
        try:
            from umap import UMAP
            reducer = UMAP(n_components=UMAP_COMPONENTS, random_state=42,
                           n_neighbors=10, min_dist=0.05)
            Xr = reducer.fit_transform(X)
        except Exception:
            Xr = PCA(n_components=min(16, X.shape[1]-1), random_state=42).fit_transform(X)
    else:
        Xr = PCA(n_components=min(16, X.shape[1]-1), random_state=42).fit_transform(X)

    # 6) kmeans(k=2)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(Xr)
    team_ids = (labels + 1).astype(int)

    # 7) write tracks parquet (same key pattern as detections)
    out = pd.DataFrame({
        "match_id": match_id,
        "object_id": [r[0] for r in rows],
        "n_samples": ns,
        "team_id": team_ids,
    }).sort_values(["team_id","object_id"])
    tracks_key = f"match_{match_id}/match_{match_id}.parquet"
    _write_parquet(BUCKET_TRACKS, tracks_key, out, cli)

    summary = out.groupby("team_id")["object_id"].count().to_dict()
    return {"match_id": match_id, "tracks": int(out.shape[0]), "by_team": summary}
