import os, io, json, tempfile, pathlib
from typing import List, Dict, Tuple
import numpy as np
import cv2
from minio import Minio
from ultralytics import YOLO


BUCKET_FRAMES     = os.getenv("BUCKET_FRAMES", "frames")
BUCKET_MODELS     = os.getenv("BUCKET_MODELS", "models")
BUCKET_HOMOGRAPHY = os.getenv("BUCKET_HOMOGRAPHY", "homography")

SEGMENT_FRAMES = int(os.getenv("SEGMENT_FRAMES", "25"))
HOMOGRAPHY_STEP = int(os.getenv("HOMOGRAPHY_STEP", "5"))
KP_CONF_TH = float(os.getenv("KP_CONF_TH", "0.5"))
PITCH_M = float(os.getenv("PITCH_M", "120"))  # meters length
PITCH_N = float(os.getenv("PITCH_N", "70"))   # meters width

# MUST match the order used during training
KEYPOINT_NAMES = [
  "corner_top_left", "left_penalty_box_top_left", "left_six_box_top_left", "left_six_box_bottom_left", "left_penalty_box_bottom_left", "corner_bottom_left",
  "left_six_box_top_right", "left_six_box_bottom_right", 
  "left_penalty_spot",
  "left_penalty_box_top_right", "left_penalty_box_center_top", "left_penalty_box_center_bottom", "left_penalty_box_bottom_right", 
  "center_top", "center_circle_top", "center_circle_bottom", "center_bottom",
  "right_penalty_box_top_left", "right_penalty_box_center_top", "right_penalty_box_center_bottom", "right_penalty_box_bottom_left",
  "right_penalty_spot",
  "right_six_box_top_left", "right_six_box_bottom_left", 
  "corner_top_right", "right_penalty_box_top_right", "right_six_box_top_right", "right_six_box_bottom_right", "right_penalty_box_bottom_right", "corner_bottom_right",
  "center_circle_left", "center_circle_right"
]


def _load_latest_model_local(cli: Minio) -> Tuple[str, str]:
    """Download latest.json and model.onnx locally; return (model_path, version)."""
    # read latest.json
    latest_key = "pitch_keypoints/latest.json"
    data = cli.get_object(BUCKET_MODELS, latest_key).read()
    latest = json.loads(data.decode("utf-8"))
    pt_path = latest.get("pt")
    onnx_path = latest.get("onnx")              # e.g., pitch_keypoints/20250101_123045/model.onnx
    version = latest.get("version","unknown")

    with tempfile.TemporaryDirectory() as td:
        if pt_path:
            dst = os.path.join(td, "model.pt")
            cli.fget_object(BUCKET_MODELS, pt_path, dst)
            # persist outside tempdir
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.close()
            import shutil; shutil.copyfile(dst, tmp.name)
            return tmp.name, version

        # fallback to onnx if pt missing
        if onnx_path:
            dst = os.path.join(td, "model.onnx")
            cli.fget_object(BUCKET_MODELS, onnx_path, dst)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
            tmp.close()
            import shutil; shutil.copyfile(dst, tmp.name)
            return tmp.name, version
    raise RuntimeError("latest.json has neither 'pt' nor 'onnx' keys")

def _list_frame_keys(match_id: int, cli: Minio) -> List[str]:
    prefix = f"match_{match_id}/"
    objs = cli.list_objects(BUCKET_FRAMES, prefix=prefix, recursive=True)
    keys = [o.object_name for o in objs if o.object_name.lower().endswith((".jpg",".jpeg",".png"))]
    keys.sort()
    return keys

def _load_image_from_minio(key: str, cli: Minio) -> np.ndarray:
    obj = cli.get_object(BUCKET_FRAMES, key)
    arr = np.frombuffer(obj.read(), np.uint8)
    obj.close(); obj.release_conn()
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _canonical_pitch_points() -> Dict[str, Tuple[float,float]]:
    """
    Canonical 2D pitch coordinates in meters (x along length 0..PITCH_M, y across width 0..PITCH_N).
    Adjust these to match your labeling semantics.
    """
    M, N = PITCH_M, PITCH_N
    return {
        # Left side corners and boxes
        "corner_top_left": (0.0, 0.0),
        "left_penalty_box_top_left": (0.0, N*0.21),  # 16.5m box
        "left_six_box_top_left": (0.0, N*0.35),      # 5.5m box
        "left_six_box_bottom_left": (0.0, N*0.65),
        "left_penalty_box_bottom_left": (0.0, N*0.79),
        "corner_bottom_left": (0.0, N),
        
        # Left side six yard box right points
        "left_six_box_top_right": (5.5, N*0.35),
        "left_six_box_bottom_right": (5.5, N*0.65),
        
        # Left penalty spot and box details
        "left_penalty_spot": (11.0, N/2.0),
        "left_penalty_box_top_right": (20, N*0.21),
        "left_penalty_box_center_top": (20, N*0.35),
        "left_penalty_box_center_bottom": (20, N*0.65),
        "left_penalty_box_bottom_right": (20, N*0.79),
        
        # Center line points
        "center_top": (M/2.0, 0.0),
        "center_circle_top": (M/2.0, N*0.4),
        "center_circle_bottom": (M/2.0, N*0.6),
        "center_bottom": (M/2.0, N),
        
        # Right side penalty box and details (mirrored from left)
        "right_penalty_box_top_left": (M-20, N*0.21),
        "right_penalty_box_center_top": (M-20, N*0.35),
        "right_penalty_box_center_bottom": (M-20, N*0.65),
        "right_penalty_box_bottom_left": (M-20, N*0.79),
        
        # Right penalty spot
        "right_penalty_spot": (M-11.0, N/2.0),
        
        # Right side six yard box left points
        "right_six_box_top_left": (M-5.5, N*0.35),
        "right_six_box_bottom_left": (M-5.5, N*0.65),
        
        # Right side corners and boxes
        "corner_top_right": (M, 0.0),
        "right_penalty_box_top_right": (M, N*0.21),
        "right_six_box_top_right": (M, N*0.35),
        "right_six_box_bottom_right": (M, N*0.65),
        "right_penalty_box_bottom_right": (M, N*0.79),
        "corner_bottom_right": (M, N),
        
        # Center circle left and right points
        "center_circle_left": (M/2.0 - N*0.1, N/2.0),   
        "center_circle_right": (M/2.0 + N*0.1, N/2.0)
    }

def _fit_homography_ransac(img_pts: np.ndarray, pitch_pts: np.ndarray):
    """
    img_pts: (N,2), pixel coords
    pitch_pts: (N,2), meters coords
    returns H (3x3), mask, inliers
    """
    # OpenCV expects float32
    src = img_pts.astype(np.float32)
    dst = pitch_pts.astype(np.float32)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.995)
    if mask is None:
        inliers = 0
    else:
        inliers = int(mask.sum())
    return H, inliers, len(src)

def estimate_homography_for_match(match_id: int, segment_frames: int, step: int, conf_th: float, cli: Minio) -> Dict:
    # 1) load model
    model_path, version = _load_latest_model_local(cli)
    model = YOLO(model_path)

    # 2) list frames
    keys = _list_frame_keys(match_id, cli)
    if not keys:
        return {"match_id": match_id, "segments": 0, "written": 0, "note": "no frames"}

    # 3) iterate segments
    written = 0
    segments = int(np.ceil(len(keys)/segment_frames))
    pitch_map = _canonical_pitch_points()

    for s in range(segments):
        # Print progress every 10%
        progress = (s + 1) / segments * 100
        if progress % 10 == 0:
            print(f"Processing segments: {int(progress)}% complete ({s + 1}/{segments})")
            
        start = s*segment_frames
        end = min((s+1)*segment_frames, len(keys)) - 1
        if end <= start: 
            continue

        # gather correspondences across the segment
        img_pts = []
        pitch_pts = []
        used_names = set()

        img_pts_by_name: Dict[str, list] = {k: [] for k in KEYPOINT_NAMES}

        for i in range(start, end+1, step):
            img = _load_image_from_minio(keys[i], cli)
            # run inference; Ultralytics outputs .keypoints for pose models
            res = model.predict(img, verbose=False)[0]
            if not hasattr(res, "keypoints") or res.keypoints is None:
                continue
            # assume single 'pitch' obj with K keypoints
            kpts = res.keypoints.xy if res.keypoints is not None else None
            confs = res.keypoints.conf if res.keypoints is not None else None
            if kpts is None or len(kpts) == 0:
                continue
            k = kpts[0].cpu().numpy()  # (K,2)
            c = confs[0].cpu().numpy() if confs is not None else np.ones((k.shape[0],), dtype=np.float32)

            K = min(len(KEYPOINT_NAMES), k.shape[0])
            for j in range(K):
                if c[j] < conf_th: 
                    continue
                name = KEYPOINT_NAMES[j]
                if name not in pitch_map:
                    continue
                x, y = float(k[j][0]), float(k[j][1])
                img_pts.append([x, y])
                pitch_pts.append(pitch_map[name])
                img_pts_by_name[name].append([x, y])
                used_names.add(name)

        if len(img_pts) < 4:
            # not enough constraints
            continue

        img_pts = np.array(img_pts)
        pitch_pts = np.array(pitch_pts)
        H, inliers, total = _fit_homography_ransac(img_pts, pitch_pts)
        if H is None:
            continue
        # aggregate representative image coords per name (median is robust)
        keypoints_img = []
        for name, pts in img_pts_by_name.items():
            if not pts:
                continue
            arr = np.asarray(pts, dtype=np.float32)
            med = np.median(arr, axis=0)  # (x, y)
            keypoints_img.append({
                "name": name,
                "x": float(med[0]),
                "y": float(med[1]),
                "n": int(len(pts)) # how many times this keypoint was seen in the segment
            })
        
        # write JSON to MinIO
        payload = {
            "match_id": match_id,
            "segment_index": s,
            "frame_start": start,
            "frame_end": end,
            "model_version": version,
            "n_points": int(total),
            "n_inliers": int(inliers),
            "inlier_ratio": float(inliers/max(total,1)),
            "H": H.tolist(),
            "keypoints_img": keypoints_img,
            "keypoints_used": sorted(list(used_names))
        }
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, f"seg_{s:03d}.json")
            with open(p, "w") as f:
                json.dump(payload, f)
            key = f"match_{match_id}/seg_{s:03d}.json"
            _ = cli.fput_object(BUCKET_HOMOGRAPHY, key, p, content_type="application/json")
        written += 1

    return {"match_id": match_id, "segments": segments, "written": written}
