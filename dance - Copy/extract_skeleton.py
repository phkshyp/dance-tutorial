## this code is the first step in this project:
##    - it takes the video (you'll have to run command "python extract_skeleton.py test_dance.mp4 skeleton.json")
##    - identifies dancer, analyses thousands of frams 
##    - maps 33 markers in each frame 
##    - outputs "skeleton.json"


"""
Extract a 3D skeleton from a video with MediaPipe BlazePose, including:
  - per-bone local rotations (quaternions) for driving a rigged avatar
  - a metric, floor-grounded root trajectory (so travel across the floor is kept)
  - foot-contact flags

Install:  pip install mediapipe opencv-python numpy scipy
Run:      python extract_skeleton.py dance.mp4 skeleton.json [--focal 0.8] [--model path.task]
Works with both current MediaPipe (Tasks API, model auto-downloaded on first run) and
older versions that still include mp.solutions.

Output coordinates (right-handed, meters):
  +x = the dancer's LEFT when they face the camera (screen right)
  +y = up, floor at y = 0
  +z = toward the viewer/camera
The dancer starts at x = z = 0. Rest pose for rotations is a T-pose facing +z.
"""
import argparse
import json

import os
import urllib.request

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

N = 33
# BlazePose landmark order (fixed by the model), so the maths below never needs mediapipe.
NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE",
    "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
I = {n: i for i, n in enumerate(NAMES)}
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

PARENT = {
    "hips": None, "chest": "hips", "head": "chest",
    "L_upperarm": "chest", "L_forearm": "L_upperarm", "L_hand": "L_forearm",
    "R_upperarm": "chest", "R_forearm": "R_upperarm", "R_hand": "R_forearm",
    "L_thigh": "hips", "L_shin": "L_thigh", "L_foot": "L_shin",
    "R_thigh": "hips", "R_shin": "R_thigh", "R_foot": "R_shin",
}


# ---------------------------------------------------------------- extraction
def make_detector(model_path):
    """
    Returns detect(rgb_frame, timestamp_ms) -> (world_landmarks, image_landmarks) or None,
    plus a close() function. Uses the Tasks API (current MediaPipe); falls back to the legacy
    mp.solutions API on older installs that still ship it.
    """
    import mediapipe as mp
    if hasattr(mp, "solutions"):  # legacy API (mediapipe <= ~0.10.21)
        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def detect(rgb, ts):
            r = pose.process(rgb)
            if not r.pose_world_landmarks:
                return None
            return r.pose_world_landmarks.landmark, r.pose_landmarks.landmark
        return detect, pose.close

    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    if not os.path.exists(model_path):
        print(f"downloading pose model to {model_path} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))

    def detect(rgb, ts):
        r = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
        if not r.pose_world_landmarks:
            return None
        return r.pose_world_landmarks[0], r.pose_landmarks[0]
    return detect, landmarker.close


def extract(video_path, model_path="pose_landmarker_heavy.task"):
    detect, close = make_detector(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    world, image, vis = [], [], []
    nan = [[np.nan] * 3] * N
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(i * 1000 / fps))
        i += 1
        if out:
            wl, il = out
            world.append([[p.x, p.y, p.z] for p in wl])
            image.append([[p.x, p.y, p.z] for p in il])
            vis.append([p.visibility for p in wl])
        else:
            world.append(nan); image.append(nan); vis.append([0.0] * N)
    cap.release(); close()
    return fps, w, h, np.array(world), np.array(image), np.array(vis)


def fill_gaps(a):
    flat = a.reshape(len(a), -1)
    t = np.arange(len(a))
    for c in range(flat.shape[1]):
        good = ~np.isnan(flat[:, c])
        if good.sum() >= 2:
            flat[:, c] = np.interp(t, t[good], flat[good, c])
    return flat.reshape(a.shape)


def smooth(a, window=9, order=3):
    return savgol_filter(a, window, order, axis=0) if len(a) >= window else a


def suspect_frames(img, w, h):
    """Frames where the tracked hips jump implausibly far: possible switch to another person."""
    scale = np.array([w, h])
    hips = img[:, [I["LEFT_HIP"], I["RIGHT_HIP"]], :2].mean(1) * scale
    torso = np.linalg.norm((img[:, I["LEFT_SHOULDER"], :2] - img[:, I["LEFT_HIP"], :2]) * scale, axis=1)
    step = np.linalg.norm(np.diff(hips, axis=0), axis=1)
    return (np.where(step > 0.6 * np.nanmedian(torso))[0] + 1).tolist()


# ------------------------------------------------------------------ rotations
def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def basis_xy(x, y):
    """Full orientation from a lateral axis (x) and an up-ish axis (y)."""
    x = unit(x); z = unit(np.cross(x, y)); y = np.cross(z, x)
    return R.from_matrix(np.stack([x, y, z], axis=-1))


def basis_xz(x, z):
    """Full orientation from a lateral axis (x) and a forward-ish axis (z)."""
    x = unit(x); y = unit(np.cross(z, x)); z = np.cross(x, y)
    return R.from_matrix(np.stack([x, y, z], axis=-1))


def arc(rest, targets):
    """Shortest rotation taking each rest direction (F x 3 or a single vector) onto each target."""
    b = unit(targets)
    a = unit(np.broadcast_to(np.asarray(rest, float), b.shape))
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=1)
    ang = np.arctan2(s, np.einsum("ij,ij->i", a, b))
    axis = axis / np.maximum(s, 1e-9)[:, None]
    anti = s < 1e-6
    if anti.any():  # parallel or exactly opposite: any perpendicular axis (angle is 0 or pi)
        ex = np.broadcast_to([1.0, 0, 0], a.shape); ey = np.broadcast_to([0, 1.0, 0], a.shape)
        helper = np.where((np.abs(a[:, [0]]) < 0.9), ex, ey)
        axis[anti] = unit(np.cross(a, helper))[anti]
    return R.from_rotvec(axis * ang[:, None])


def swing(parent_world, rest_dir, targets):
    """
    World rotation of a bone that points at `targets`, inheriting the parent's twist.
    The rest direction is first expressed in the parent's current frame, so the result only
    adds the minimal swing needed. This avoids spurious twist along the bone, which would
    otherwise show up as candy-wrapper skinning and a wrongly oriented knee/elbow hinge.
    """
    rest_world = parent_world.apply(np.asarray(rest_dir, float))
    return arc(rest_world, targets) * parent_world


def world_rotations(P):
    """P: frames x 33 x 3 (y up, z toward viewer, dancer's left = +x). Returns bone -> world Rotation."""
    g = lambda n: P[:, I[n]]
    mid = lambda a, b: (g(a) + g(b)) / 2
    spine = mid("LEFT_SHOULDER", "RIGHT_SHOULDER") - mid("LEFT_HIP", "RIGHT_HIP")
    ear_c = mid("LEFT_EAR", "RIGHT_EAR")
    W = {
        "hips": basis_xy(g("LEFT_HIP") - g("RIGHT_HIP"), spine),
        "chest": basis_xy(g("LEFT_SHOULDER") - g("RIGHT_SHOULDER"), spine),
        "head": basis_xz(g("LEFT_EAR") - g("RIGHT_EAR"), g("NOSE") - ear_c),
    }
    for tag, side, sx in (("L", "LEFT", 1), ("R", "RIGHT", -1)):
        arm = [sx, 0, 0]; down = [0, -1, 0]
        W[f"{tag}_upperarm"] = swing(W["chest"], arm, g(f"{side}_ELBOW") - g(f"{side}_SHOULDER"))
        W[f"{tag}_forearm"] = swing(W[f"{tag}_upperarm"], arm, g(f"{side}_WRIST") - g(f"{side}_ELBOW"))
        knuckles = (g(f"{side}_INDEX") + g(f"{side}_PINKY")) / 2
        W[f"{tag}_hand"] = swing(W[f"{tag}_forearm"], arm, knuckles - g(f"{side}_WRIST"))
        W[f"{tag}_thigh"] = swing(W["hips"], down, g(f"{side}_KNEE") - g(f"{side}_HIP"))
        W[f"{tag}_shin"] = swing(W[f"{tag}_thigh"], down, g(f"{side}_ANKLE") - g(f"{side}_KNEE"))
        W[f"{tag}_foot"] = swing(W[f"{tag}_shin"], [0, 0, 1], g(f"{side}_FOOT_INDEX") - g(f"{side}_HEEL"))
    return W


def local_quats(W):
    out = {}
    for bone, parent in PARENT.items():
        rot = W[bone] if parent is None else W[parent].inv() * W[bone]
        q = rot.as_quat()  # x, y, z, w
        for i in range(1, len(q)):          # keep sign continuous between frames
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        if len(q) >= 9:
            q = savgol_filter(q, 9, 3, axis=0)
        out[bone] = (q / np.linalg.norm(q, axis=1, keepdims=True))
    return out


# ------------------------------------------------------- root path and contacts
def root_trajectory(P, img, w, h, focal_factor):
    """
    Pinhole estimate of the hip-center position in meters.
    Depth comes from comparing metric limb lengths (projected onto the image plane,
    so foreshortening cancels) with the same limbs measured in pixels.
    """
    f = focal_factor * max(w, h)
    uv = img[..., :2] * np.array([w, h])
    pairs = [("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_HIP", "RIGHT_HIP"),
             ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
             ("LEFT_HIP", "LEFT_KNEE"), ("RIGHT_HIP", "RIGHT_KNEE")]
    metric = sum(np.linalg.norm(P[:, I[a], :2] - P[:, I[b], :2], axis=1) for a, b in pairs)
    pixels = sum(np.linalg.norm(uv[:, I[a]] - uv[:, I[b]], axis=1) for a, b in pairs)
    Z = f * metric / np.maximum(pixels, 1e-6)
    Z = savgol_filter(Z, 15, 2) if len(Z) >= 15 else Z

    hip = (uv[:, I["LEFT_HIP"]] + uv[:, I["RIGHT_HIP"]]) / 2
    X = (hip[:, 0] - w / 2) * Z / f
    Y = -(hip[:, 1] - h / 2) * Z / f
    root = np.stack([X, Y, -Z], axis=1)          # camera looks down -z
    root = smooth(root, 15, 2)

    root[:, [0, 2]] -= root[0, [0, 2]]           # dancer starts at x = z = 0
    feet = ["LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]
    foot_h = np.stack([root[:, 1] + P[:, I[n], 1] for n in feet], axis=1)
    root[:, 1] -= np.percentile(foot_h.min(1), 2)  # lowest foot position = floor
    return root


def foot_contacts(P, root, fps):
    res = []
    for side in ("LEFT", "RIGHT"):
        pts = np.stack([P[:, I[f"{side}_HEEL"]], P[:, I[f"{side}_FOOT_INDEX"]]], 1) + root[:, None]
        k = pts[..., 1].argmin(1)
        low = pts[np.arange(len(pts)), k]
        speed = np.r_[0, np.linalg.norm(np.diff(low[:, [0, 2]], axis=0), axis=1)] * fps
        res.append((low[:, 1] < 0.08) & (speed < 0.6))
    return np.stack(res, 1)


# ----------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--focal", type=float, default=0.8,
                    help="focal length as a fraction of the longer image side (phones ~0.7-0.9)")
    ap.add_argument("--model", default="pose_landmarker_heavy.task",
                    help="pose model file; downloaded automatically if missing (new MediaPipe only)")
    a = ap.parse_args()

    fps, w, h, world, image, vis = extract(a.video, a.model)
    missing = int(np.isnan(world[:, 0, 0]).sum())
    jumps = suspect_frames(image, w, h)
    print(f"{len(world)} frames @ {fps:.1f} fps, {missing} without a detection")
    if jumps:
        print(f"WARNING: tracking jumped at frames {jumps[:10]}{'...' if len(jumps) > 10 else ''} "
              "(possibly locked onto another person). Crop to the main dancer and re-run.")

    world = smooth(fill_gaps(world))
    image = smooth(fill_gaps(image))
    world[..., 1] *= -1   # MediaPipe y is down  -> up
    world[..., 2] *= -1   # MediaPipe z is away  -> toward viewer (keeps right-handed)

    root = root_trajectory(world, image, w, h, a.focal)
    quats = local_quats(world_rotations(world))
    contacts = foot_contacts(world, root, fps)

    json.dump({
        "fps": fps,
        "hierarchy": PARENT,
        "rest_pose": "T-pose facing +z, arms along +/-x, legs down -y; L side = +x",
        "quat_order": "x,y,z,w",
        "root_position": root.round(4).tolist(),                      # frames x 3, meters
        "rotations": {b: q.round(5).tolist() for b, q in quats.items()},  # bone -> frames x 4 (local)
        "foot_contact": contacts.astype(int).tolist(),                # frames x [L, R]
        "landmarks_world": world.round(4).tolist(),                   # frames x 33 x 3, hip-centered
        "visibility": vis.round(3).tolist(),
        "suspect_frames": jumps,
    }, open(a.out, "w"))
    print("saved", a.out)