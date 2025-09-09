import os, json, tempfile, pathlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union

PITCH_M = float(os.getenv("PITCH_M", "120"))   
PITCH_N = float(os.getenv("PITCH_N", "70"))    
FPS = int(os.getenv("FRAMES_FPS", "25")) 
H_TARGET = os.getenv("HOMOGRAPHY_TARGET", "normalized")          

PLAYER_CLASSES = {"player", "goalkeeper"}  # exclude referee for control/possession

# -------- Helpers: MinIO/Parquet --------
def _first_parquet_key(s3, bucket: str, prefix: str) -> str:
    for o in s3.list_objects(bucket, prefix=prefix, recursive=True):
        if o.object_name.endswith(".parquet"):
            return o.object_name
    raise FileNotFoundError(f"No parquet under s3://{bucket}/{prefix}")

def _load_parquet(s3, bucket: str, key: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, pathlib.Path(key).name)
        s3.fget_object(bucket, key, dst)
        return pd.read_parquet(dst, engine="pyarrow")

def load_detections_df(s3, bucket_detections: str, match_id: int) -> pd.DataFrame:
    key = _first_parquet_key(s3, bucket_detections, f"match_{match_id}/")
    df = _load_parquet(s3, bucket_detections, key)
    # normalize columns
    exp = ["frame_id","filename","class_name","conf","x1","y1","x2","y2","object_id"]
    for c in exp:
        if c not in df.columns: df[c] = None
    return df

def load_team_map(s3, bucket_tracks: str, match_id: int) -> Dict[int, int]:
    """Return {object_id -> team_id} (0/1). Requires team assignment parquet."""
    key = _first_parquet_key(s3, bucket_tracks, f"match_{match_id}/")
    df = _load_parquet(s3, bucket_tracks, key)
    # Try common column names
    team_col = None
    for cand in ["team_id", "team", "assigned_team"]:
        if cand in df.columns:
            team_col = cand; break
    if team_col is None or "object_id" not in df.columns:
        raise RuntimeError("Tracks parquet missing team mapping (need columns: object_id + team_id). Run /teams/assign first.")
    # Build last known team per object_id
    df = df[["object_id", team_col]].dropna()
    df[team_col] = df[team_col].astype(int)
    return df.drop_duplicates(subset=["object_id"], keep="last").set_index("object_id")[team_col].to_dict()

# -------- Homography handling --------
def load_homographies(s3, bucket_homog: str, match_id: int) -> List[dict]:
    prefix = f"match_{match_id}/"
    objs = list(s3.list_objects(bucket_homog, prefix=prefix, recursive=True))
    if not objs:
        raise FileNotFoundError("no homography found; run /homography/estimate")
    items = []
    with tempfile.TemporaryDirectory() as td:
        for o in objs:
            dst = os.path.join(td, pathlib.Path(o.object_name).name)
            s3.fget_object(bucket_homog, o.object_name, dst)
            with open(dst, "r") as f:
                items.append(json.load(f))
    items.sort(key=lambda x: x.get("segment_index", 0))
    return items

def _pick_H_for_frame(hlist: List[dict], frame_id: int) -> Optional[np.ndarray]:
    # Find segment covering this frame; else nearest by segment_index
    for it in hlist:
        s, e = it.get("frame_start", 0), it.get("frame_end", 0)
        if s <= frame_id <= e:
            H = np.array(it["H"], dtype=float)
            return H
    # Fallback: choose first available
    if hlist:
        return np.array(hlist[0]["H"], dtype=float)
    return None

def _apply_H(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return np.nan, np.nan
    xn, yn = p[0]/p[2], p[1]/p[2]
    return float(xn), float(yn)

# -------- Positions (projection) --------
def compute_positions_df(
    s3, match_id: int,
    bucket_detections: str,
    bucket_tracks: str,
    bucket_homog: str,
    bottom_center: bool = True,
) -> pd.DataFrame:
    det = load_detections_df(s3, bucket_detections, match_id)
    team_map = load_team_map(s3, bucket_tracks, match_id)
    hlist = load_homographies(s3, bucket_homog, match_id)

    # compute bbox anchor
    x_c = (det["x1"].astype(float) + det["x2"].astype(float)) / 2.0
    y_c = det["y2"].astype(float) if bottom_center else (det["y1"].astype(float) + det["y2"].astype(float))/2.0
    det["ax"] = x_c
    det["ay"] = y_c
    xs, ys, x_m, y_m, team_ids = [], [], [], [], []
    for row in det.itertuples(index=False):
        H = _pick_H_for_frame(hlist, int(getattr(row, "frame_id")))
        if H is None:
            xs.append(np.nan); ys.append(np.nan); x_m.append(np.nan); y_m.append(np.nan); team_ids.append(None)
            continue
        xn, yn = _apply_H(H, float(getattr(row, "ax")), float(getattr(row, "ay")))
        if H_TARGET == "meters":
            x_unit, y_unit = float(xn/PITCH_M), float(yn/PITCH_N)
            x_m_val, y_m_val = float(xn), float(yn)
        else:
            x_unit, y_unit = float(xn), float(yn)
            x_m_val, y_m_val = x_unit * PITCH_M, y_unit * PITCH_N
        xs.append(x_unit); ys.append(y_unit)
        x_m.append(x_m_val); y_m.append(y_m_val)
        oid = getattr(row, "object_id")
        team_ids.append(int(team_map.get(int(oid), -1)) if pd.notna(oid) else None)

    out = det.copy()
    out["x_norm"], out["y_norm"] = xs, ys
    out["x_m"], out["y_m"] = x_m, y_m
    out["team_id"] = team_ids
    return out

# -------- Possession (ball nearest) --------
def possession_timeseries(
    pos_df: pd.DataFrame,
    max_dist_m: float = 4.0,
    delta_margin_m: float = 0.5,
    fps: int = FPS,
) -> Tuple[pd.DataFrame, dict]:
    df = pos_df.copy()
    # Separate
    ball = df[df["class_name"]=="ball"][["frame_id","x_m","y_m"]].dropna()
    ply  = df[df["class_name"].isin(PLAYER_CLASSES)][["frame_id","object_id","team_id","x_m","y_m"]].dropna(subset=["x_m","y_m"])
    # ffill ball when missing (short gaps)
    ball = ball.sort_values("frame_id").drop_duplicates("frame_id", keep="first").set_index("frame_id")
    ball = ball.reindex(range(int(df["frame_id"].min()), int(df["frame_id"].max())+1))    
    ball[["x_m","y_m"]] = ball[["x_m","y_m"]].ffill(limit=int(fps*2))  # up to ~2s gap
    # Compute nearest player / team per frame
    series = []
    last_team = None
    for f, b in ball.dropna().iterrows():
        players = ply[ply["frame_id"]==f]
        if players.empty:
            series.append((f, None)); continue
        d = np.sqrt((players["x_m"]-b["x_m"])**2 + (players["y_m"]-b["y_m"])**2)
        idxmin = d.idxmin(); mind = float(d.loc[idxmin])
        team = int(players.loc[idxmin, "team_id"]) if pd.notna(players.loc[idxmin, "team_id"]) else None
        if team is None or mind > max_dist_m:
            series.append((f, last_team))
            continue
        # Hysteresis vs the other team
        other = 1 - team
        other_min = float(d[players["team_id"]==other].min()) if (players["team_id"]==other).any() else 1e9
        if other_min - mind < delta_margin_m and last_team is not None:
            # not decisively closer: keep last
            team = last_team
        series.append((f, team))
        last_team = team

    ts = pd.DataFrame(series, columns=["frame_id","team"])
    ts["time_s"] = ts["frame_id"] / fps
    # Replace NaN with None for JSON compatibility
    ts = ts.where(pd.notnull(ts), None)

    # Summary
    counts = ts["team"].value_counts(dropna=True).to_dict()
    total = int(ts["team"].notna().sum())
    pct = {int(k): round(v*100.0/total, 2) for k,v in counts.items()} if total else {}
    summary = {"frames_counted": total, "percent_by_team": pct}
    return ts, summary

# -------- Voronoi control areas --------
def _finite_polygons_2d(vor: Voronoi, radius: float = 1e6):
    """
    Reconstruct infinite Voronoi regions to finite polygons.
    Returns (regions, vertices) where each region is a list of vertex indices.
    Adapted from scipy cookbook.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    # Map ridge points to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            # Finite
            new_regions.append(vertices)
            continue

        # Reconstruct infinite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v1 if v1 >= 0 else v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        # Order the region's vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

def _voronoi_polygons(points: np.ndarray, bbox_rect: Polygon) -> list[Polygon]:
    """
    Finite Voronoi cells clipped to bbox.
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [bbox_rect]  # one point controls entire pitch

    vor = Voronoi(points)
    regions, vertices = _finite_polygons_2d(vor)
    polys = []
    for region in regions:
        poly = Polygon(vertices[region])
        polys.append(poly.intersection(bbox_rect))
    return polys

def control_area_timeseries(
    pos_df: pd.DataFrame,
    stride: int = FPS,  # ~1 per second
) -> pd.DataFrame:
    df = pos_df[pos_df["class_name"].isin(PLAYER_CLASSES)].dropna(subset=["x_m","y_m","team_id"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["frame_id","time_s","team1_area_pct","team2_area_pct"])
    frames = sorted(df["frame_id"].unique().tolist())
    frames = frames[::max(1, int(stride))]
    rows = []
    # Always use the actual pitch dimensions since x_m and y_m are in meters
    bbox = box(0.0, 0.0, PITCH_M, PITCH_N)
    
    for f in frames:
        snap = df[df["frame_id"]==f]
        if snap.empty or len(snap) < 2:
            continue
        pts = snap[["x_m","y_m"]].to_numpy()
        polys = _voronoi_polygons(pts, bbox)
        # assign polygon areas to each point/team
        areas_by_team = {1:0.0, 2:0.0}  # Teams are 1 and 2
        for (poly, (_, row)) in zip(polys, snap.iterrows()):
            a = max(poly.area, 0.0)
            t = int(row["team_id"]) if row["team_id"] in [1,2] else None
            if t is not None:
                areas_by_team[t] = areas_by_team.get(t,0.0) + a
        total_area = float(bbox.area)
        t1 = areas_by_team.get(1,0.0) / total_area * 100.0  # Team 1
        t2 = areas_by_team.get(2,0.0) / total_area * 100.0  # Team 2
        rows.append((f, f/FPS, round(t1,2), round(t2,2)))  # Removed extra *100
    out = pd.DataFrame(rows, columns=["frame_id","time_s","team1_area_pct","team2_area_pct"])
    return out

# -------- Momentum (smoothed territory advantage) --------
def momentum_timeseries(ctrl_df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    if ctrl_df.empty:
        return ctrl_df
    df = ctrl_df.copy().sort_values("frame_id")
    # Momentum index = team area share (0..100) smoothed; provide both teams
    df["team1_momentum"] = df["team1_area_pct"].ewm(alpha=alpha).mean()
    df["team2_momentum"] = df["team2_area_pct"].ewm(alpha=alpha).mean()
    return df[["frame_id","time_s","team1_momentum","team2_momentum"]]
