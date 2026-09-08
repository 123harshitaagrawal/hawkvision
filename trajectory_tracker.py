import os
import math
import time
import subprocess
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from speed_tracker import CricketSpeedTracker

# Offline Haar Cascade face detection initialization
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_HAAR_PATH = os.path.join(_BASE_DIR, "haarcascade_frontalface_default.xml")
_FACE_CASCADE = None
if os.path.exists(_HAAR_PATH):
    try:
        _FACE_CASCADE = cv2.CascadeClassifier(_HAAR_PATH)
        if _FACE_CASCADE.empty():
            _FACE_CASCADE = None
    except Exception:
        _FACE_CASCADE = None

def detect_faces_fast(frame, target_w=160, target_h=90):
    """
    Run fast multi-scale face detection on downscaled grayscale frame.
    Returns list of (fx, fy, fw, fh) scaled to original frame dimensions.
    """
    if _FACE_CASCADE is None or frame is None:
        return []
    try:
        h, w = frame.shape[:2]
        small_gray = cv2.cvtColor(cv2.resize(frame, (target_w, target_h)), cv2.COLOR_BGR2GRAY)
        scale_x = w / float(target_w)
        scale_y = h / float(target_h)
        faces = _FACE_CASCADE.detectMultiScale(small_gray, scaleFactor=1.18, minNeighbors=3, minSize=(14, 14))
        return [(int(fx * scale_x), int(fy * scale_y), int(fw * scale_x), int(fh * scale_y)) for (fx, fy, fw, fh) in faces]
    except Exception:
        return []

def get_distance_to_box(cx, cy, box):
    """
    Euclidean distance from point (cx, cy) to edge of box [x1, y1, x2, y2].
    Returns 0.0 if (cx, cy) is inside the box.
    Returns float('inf') if box is None.
    """
    if box is None:
        return float('inf')
    x1, y1, x2, y2 = box
    dx = max(x1 - cx, 0.0, cx - x2)
    dy = max(y1 - cy, 0.0, cy - y2)
    return math.hypot(dx, dy)

def get_distance_to_nearest_person(cx, cy, person_boxes):
    """Euclidean distance from (cx, cy) to edge of nearest person box."""
    if not person_boxes:
        return float('inf')
    return min(get_distance_to_box(cx, cy, b) for b in person_boxes)

def identify_bowler_and_batsman(person_boxes, bowler_release_zone, width, height):
    """
    Classify detected persons into (bowler_box, batsman_box) based on release zone.
    Bowler is the person nearest to the release zone.
    Batsman is the person furthest down-pitch from release zone.
    """
    if not person_boxes:
        return None, None
    if len(person_boxes) == 1:
        return person_boxes[0], None
    
    if bowler_release_zone is not None:
        rx, ry = bowler_release_zone
        closest_idx = -1
        min_dist = float('inf')
        for idx, box in enumerate(person_boxes):
            d = get_distance_to_box(rx, ry, box)
            if d < min_dist:
                min_dist = d
                closest_idx = idx
        bowler_box = person_boxes[closest_idx]
        other_boxes = [b for i, b in enumerate(person_boxes) if i != closest_idx]
        batsman_box = max(other_boxes, key=lambda b: get_distance_to_box(rx, ry, b)) if other_boxes else None
        return bowler_box, batsman_box
    else:
        return person_boxes[0], (person_boxes[1] if len(person_boxes) > 1 else None)

def angle_between_lines(m1, m2=1):
    """Calculate the angle between two lines."""
    if m1 != -1 / m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

def create_bezier_curve(points, smoothness=30):
    """Smooth points using quadratic/cubic Bezier curve."""
    if len(points) < 3:
        return np.array(points, dtype=np.int32)
    t = np.linspace(0, 1, smoothness)
    curve = []
    p0, p1, p2 = points[0], points[1], points[2]
    for val in t:
        x = (1 - val) ** 2 * p0[0] + 2 * (1 - val) * val * p1[0] + val ** 2 * p2[0]
        y = (1 - val) ** 2 * p0[1] + 2 * (1 - val) * val * p1[1] + val ** 2 * p2[1]
        curve.append([int(x), int(y)])
    return np.array(curve, dtype=np.int32)

def convert_to_browser_h264(input_path, output_path):
    """
    Convert video to browser-standard H.264 (yuv420p + faststart)
    using imageio_ffmpeg or system ffmpeg for full browser compatibility.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        "-crf", "22",
        output_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0
    except Exception as e:
        print(f"H.264 conversion warning: {e}")
        return False

def pass_a_coarse_windows(video_path, pre_pad_sec=0.5, post_pad_sec=0.5, progress_callback=None):
    """
    Pass A: Coarse candidate delivery window detector using downsampled frame differencing.
    Fast, cheap scan across the video to find bursts of motion separated by dead time.
    """
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    if total_frames <= int(fps * 3.2):
        cap.release()
        return [(0, total_frames)]

    FLOW_W, FLOW_H = 160, 90
    prev_corridor = None
    motion_scores = []
    frame_idx = 0
    phone_cutoff_frame = total_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        small = cv2.resize(frame, (FLOW_W, FLOW_H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corridor = gray[int(FLOW_H * 0.12):int(FLOW_H * 0.92), int(FLOW_W * 0.12):int(FLOW_W * 0.88)]

        if prev_corridor is not None:
            diff = float(np.mean(cv2.absdiff(corridor, prev_corridor)))
            if frame_idx > total_frames * 0.75 and diff > 14.0 and phone_cutoff_frame == total_frames:
                phone_cutoff_frame = max(0, frame_idx - int(fps * 0.5))
            motion_scores.append(min(diff, 8.0))
        else:
            motion_scores.append(0.0)

        prev_corridor = corridor

        if progress_callback and (frame_idx % 30 == 0 or frame_idx >= total_frames):
            try:
                progress_callback(frame_idx, total_frames, 0.0, False,
                                  "Pass A: Scanning video for candidate delivery windows...")
            except Exception:
                pass

    cap.release()

    if not motion_scores:
        return [(0, total_frames)]

    effective_end = min(total_frames, phone_cutoff_frame)
    scores_arr = np.array(motion_scores[:effective_end], dtype=np.float32)

    k_size = max(5, int(fps * 0.35))
    if k_size % 2 == 0:
        k_size += 1
    smooth = np.convolve(scores_arr, np.ones(k_size) / k_size, mode='same')

    p50 = float(np.percentile(smooth, 50))
    th = max(1.5, min(p50, 2.2))

    peaks = [i for i in range(1, len(smooth) - 1) if smooth[i] >= th and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    if not peaks:
        return [(0, total_frames)]

    gap_thresh = int(fps * 0.5)
    groups = []
    cur_group = [peaks[0]]
    for p in peaks[1:]:
        if p - cur_group[-1] <= gap_thresh:
            cur_group.append(p)
        else:
            groups.append((cur_group[0], cur_group[-1]))
            cur_group = [p]
    if cur_group:
        groups.append((cur_group[0], cur_group[-1]))

    # Connect bowler release & batsman stroke if gap <= 4.0s and combined span <= 8.5s
    deliveries = []
    for g in groups:
        if not deliveries:
            deliveries.append(g)
        else:
            prev_s, prev_e = deliveries[-1]
            prev_dur = (prev_e - prev_s) / fps
            gap_sec = (g[0] - prev_e) / fps
            comb_sec = (g[1] - prev_s) / fps
            if prev_dur < 3.0 and gap_sec <= 4.0 and comb_sec <= 8.5:
                deliveries[-1] = (prev_s, g[1])
            else:
                deliveries.append(g)

    pre_pad = int(fps * pre_pad_sec)
    post_pad = int(fps * post_pad_sec)
    coarse_windows = []
    for s, e in deliveries:
        w_s = max(0, s - pre_pad)
        w_e = min(effective_end, e + post_pad)
        if (w_e - w_s) >= int(fps * 1.5):
            coarse_windows.append((w_s, w_e))

    for i in range(len(coarse_windows) - 1):
        if coarse_windows[i][1] > coarse_windows[i + 1][0]:
            mid = (coarse_windows[i][1] + coarse_windows[i + 1][0]) // 2
            coarse_windows[i] = (coarse_windows[i][0], mid)
            coarse_windows[i + 1] = (mid, coarse_windows[i + 1][1])

    return coarse_windows


def fit_parabola_r2(frames, ys):
    """Calculate R2 score for 2nd-degree polynomial fit y(t) = a*t^2 + b*t + c."""
    if len(frames) < 3:
        return 0.0
    try:
        t = np.array(frames, dtype=np.float32)
        y = np.array(ys, dtype=np.float32)
        p = np.polyfit(t, y, 2)
        y_pred = np.polyval(p, t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return max(0.0, float(1.0 - (ss_res / ss_tot)))
    except Exception:
        return 0.0


def is_delivery_candidate_cluster(cluster, width, height, fps):
    """
    Evaluates whether a candidate detection cluster matches genuine cricket ball kinematics:
    1. Total displacement span >= max(40px, 0.12 * max(width, height))
    2. Bounding box size ceiling: mean_bw <= 0.10*w, mean_bh <= 0.12*h (rejects face/body)
    3. Directional monotonicity: <= 35% direction reversals (rejects oscillating head/hands)
    4. Minimum implied speed floor: peak segment speed >= 15.0 km/h
    Returns (is_valid: bool, reason: str).
    """
    if len(cluster) < 3:
        return False, "Cluster has fewer than 3 detections"

    x_list = [d['cx'] for d in cluster]
    y_list = [d['cy'] for d in cluster]

    span_x = max(x_list) - min(x_list)
    span_y = max(y_list) - min(y_list)
    total_span = max(span_x, span_y)

    # Gate 1: Total displacement span across delivery
    min_span_allowed = max(40.0, 0.12 * max(width, height))
    if total_span < min_span_allowed:
        return False, f"Total displacement span {total_span:.1f}px < {min_span_allowed:.1f}px (head/hand nod, not delivery)"

    # Gate 2: Bounding box size ceiling relative to frame
    mean_bw = sum(d['box'][2] - d['box'][0] for d in cluster) / len(cluster)
    mean_bh = sum(d['box'][3] - d['box'][1] for d in cluster) / len(cluster)
    if mean_bw > width * 0.10 or mean_bh > height * 0.12:
        return False, f"Mean bbox size {mean_bw:.1f}x{mean_bh:.1f} exceeds ball ceiling (face/body/bat)"

    # Gate 3: Directional monotonicity along dominant axis
    dominant_coords = y_list if span_y >= span_x else x_list
    reversals = 0
    steps = 0
    for k in range(len(dominant_coords) - 2):
        d1 = dominant_coords[k + 1] - dominant_coords[k]
        d2 = dominant_coords[k + 2] - dominant_coords[k + 1]
        if abs(d1) > 2.0 and abs(d2) > 2.0:
            steps += 1
            if (d1 * d2) < 0:
                reversals += 1
    if steps >= 3 and (reversals / steps) > 0.35:
        return False, f"Reversal rate {reversals}/{steps} ({reversals/steps*100:.0f}%) exceeds 35% (oscillating motion)"

    # Gate 4: Minimum implied speed floor
    m_per_px_est = 18.2 / max(height * 0.55, 60.0)
    max_seg_speed_kmh = 0.0
    for k in range(len(cluster) - 1):
        df = max(cluster[k + 1]['frame'] - cluster[k]['frame'], 1)
        d_dist = math.hypot(cluster[k + 1]['cx'] - cluster[k]['cx'], cluster[k + 1]['cy'] - cluster[k]['cy'])
        dt_sec = df / max(fps, 1.0)
        if dt_sec > 0:
            seg_spd = (d_dist * m_per_px_est / dt_sec) * 3.6
            if seg_spd > max_seg_speed_kmh:
                max_seg_speed_kmh = seg_spd
    if max_seg_speed_kmh < 15.0:
        return False, f"Peak implied speed {max_seg_speed_kmh:.1f} km/h < min 15.0 km/h"

    return True, "Valid delivery candidate cluster"


def pass_b_refine_window(cap, model, c_start, c_end, fps, conf_thresh=0.15):
    """
    Pass B: Precise release & impact boundary refinement using YOLO ball tracking and projectile kinematics.
    Sets start_frame = ball leaving hand (-4 safety buffer) and end_frame = pitch impact/stumps (+6 safety buffer).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, c_start)
    raw_detections = []

    for f in range(c_start, c_end):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        best = None
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                if best is None or conf > best['conf']:
                    best = {'frame': f, 'cx': cx, 'cy': cy, 'conf': conf, 'box': xyxy}
        if best:
            raw_detections.append(best)

    if len(raw_detections) < 3:
        return [], False

    # Filter stationary noise
    moving_detections = []
    for i in range(len(raw_detections)):
        det = raw_detections[i]
        is_moving = True
        if i > 0:
            prev = raw_detections[i - 1]
            dt = det['frame'] - prev['frame']
            if 0 < dt <= 3:
                dist = math.hypot(det['cx'] - prev['cx'], det['cy'] - prev['cy'])
                if (dist / dt) < 1.0:
                    is_moving = False
        if is_moving:
            moving_detections.append(det)

    if len(moving_detections) < 3:
        return [], False

    # Group into delivery clusters: cricket ball flight is 0.4s-1.0s.
    # Adaptive gap threshold scaling with actual fps to avoid truncating indoor nets footage
    delivery_clusters = []
    cur_cluster = [moving_detections[0]]
    cluster_gap_thresh = max(16, int(fps * 0.90))
    for d in moving_detections[1:]:
        if d['frame'] - cur_cluster[-1]['frame'] <= cluster_gap_thresh:
            cur_cluster.append(d)
        else:
            if len(cur_cluster) >= 4:
                delivery_clusters.append(cur_cluster)
            cur_cluster = [d]
    if len(cur_cluster) >= 4:
        delivery_clusters.append(cur_cluster)

    if not delivery_clusters:
        return [], False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    refined = []
    rejected_count = 0

    for cluster_idx, cluster in enumerate(delivery_clusters, 1):
        f_list = [d['frame'] for d in cluster]
        y_list = [d['cy'] for d in cluster]
        conf_list = [d['conf'] for d in cluster]

        is_valid, reason = is_delivery_candidate_cluster(cluster, width, height, fps)
        if not is_valid:
            print(f"[Gating] Cluster {cluster_idx} rejected: {reason}")
            rejected_count += 1
            continue

        r2 = fit_parabola_r2(f_list, y_list)

        # Release frame: ball leaving bowler's hand
        first_f = f_list[0]
        release_f = max(c_start, first_f - 4)

        # Impact / end frame: ground pitch contact or stumps hit
        last_f = f_list[-1]
        impact_f = min(c_end, last_f + 3)

        # Minimum duration guarantee (>= 0.9s is sufficient for full delivery flight)
        if (impact_f - release_f) < int(fps * 0.9):
            impact_f = min(c_end, release_f + int(fps * 0.9))

        det_density = min(1.0, len(cluster) / max(1, (impact_f - release_f) * 0.55))
        mean_conf = float(np.mean(conf_list)) if conf_list else 0.5
        seg_conf = round(0.40 * r2 + 0.35 * det_density + 0.25 * mean_conf, 2)
        seg_conf = max(0.45, min(0.99, seg_conf))

        refined.append({
            "start_frame": int(release_f),
            "end_frame": int(impact_f),
            "flight_start_frame": int(first_f),
            "flight_end_frame": int(last_f),
            "detected_points": len(cluster),
            "parabola_r2": round(r2, 3),
            "confidence": seg_conf
        })

    had_rejected = (rejected_count > 0 and len(refined) == 0)
    return refined, had_rejected


def should_fallback_to_coarse_window(r, coarse_start, coarse_end, fps):
    """
    Evaluates whether a refined delivery window should fall back to its parent Pass A coarse window.
    Only falls back when the refined result is both short AND low-quality (e.g. indoor nets with sporadic tracking).
    If the refined delivery has high confidence, good parabola fit, and sufficient detected points,
    the tight window is trusted as-is (e.g. talk-shot-talk clip trimming out talking before/after).
    """
    coarse_duration = max(1, coarse_end - coarse_start)
    refined_duration = max(1, r.get('end_frame', coarse_end) - r.get('start_frame', coarse_start))

    is_low_quality = (
        r.get("confidence", 1.0) < 0.55
        or r.get("parabola_r2", 1.0) < 0.35
        or r.get("detected_points", 999) < 5
    )

    is_short = (refined_duration < 0.35 * coarse_duration and refined_duration < fps * 1.2)
    return is_short and is_low_quality


def segment_deliveries(video_path, model=None, conf_thresh=0.15, task_id='session', debug=False, progress_callback=None):
    """
    Two-Stage Multi-Delivery Auto-Segmentation Pipeline.
    Pass A: Coarse motion candidate windows.
    Pass B: Precise release-to-impact boundary refinement using YOLO ball projectile kinematics.
    Returns standardized JSON contract matching Part 1.
    """
    if not os.path.exists(video_path):
        return {"task_id": task_id, "video_filename": os.path.basename(video_path), "shots": [], "shot_count": 0}

    # Resolve YOLO model
    if model is None:
        possible_paths = [
            os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'),
            os.path.join('Cricket-Ball-Trajectory-Prediction-master', 'runs', 'detect', 'train5', 'weights', 'best.pt'),
            'yolov8s.pt'
        ]
        m_path = next((p for p in possible_paths if os.path.exists(p)), 'yolov8s.pt')
        model = YOLO(m_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Pass A: Coarse Windows
    coarse = pass_a_coarse_windows(video_path, progress_callback=progress_callback)

    all_refined = []
    any_rejected = False
    total_coarse = len(coarse)
    for idx, (cs, ce) in enumerate(coarse, 1):
        if progress_callback:
            try:
                progress_callback(idx, total_coarse, 0.0, False,
                                  f"Pass B: Refining release/impact boundaries for shot {idx} of {total_coarse}...")
            except Exception:
                pass
        out = pass_b_refine_window(cap, model, cs, ce, fps, conf_thresh=conf_thresh)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            res, had_rejected = out
        elif isinstance(out, (list, tuple)) and len(out) == 1:
            res, had_rejected = out[0], False
        else:
            res, had_rejected = out, False

        if had_rejected:
            any_rejected = True

        for r in (res or []):
            if should_fallback_to_coarse_window(r, cs, ce, fps):
                refined_dur = r.get('end_frame', ce) - r.get('start_frame', cs)
                print(f"[Segmentation] Refined window {r.get('start_frame')}-{r.get('end_frame')} is short ({refined_dur}f) and low quality (conf={r.get('confidence')}, r2={r.get('parabola_r2')}, pts={r.get('detected_points')}). Falling back to coarse window {cs}-{ce}.")
                all_refined.append({
                    "start_frame": cs,
                    "end_frame": ce,
                    "flight_start_frame": r.get("flight_start_frame", cs),
                    "flight_end_frame": r.get("flight_end_frame", ce),
                    "detected_points": r.get("detected_points", 0),
                    "parabola_r2": r.get("parabola_r2", 0.0),
                    "confidence": r.get("confidence", 0.50)
                })
            else:
                all_refined.append(r)

    cap.release()

    # Fallback to Pass A coarse windows ONLY if Pass B did NOT actively reject non-cricket movement
    # (e.g. sparse detections from motion blur), but NOT when motion was identified as non-delivery/face movement.
    if not all_refined and coarse and not any_rejected:
        for cs, ce in coarse:
            all_refined.append({
                "start_frame": cs,
                "end_frame": ce,
                "flight_start_frame": cs,
                "flight_end_frame": ce,
                "detected_points": 0,
                "parabola_r2": 0.0,
                "confidence": 0.50
            })
    elif not all_refined and any_rejected:
        print("[Segmentation] All candidate windows rejected by Pass B projectile gates (talking face / non-delivery). Zero shots produced.")

    # Midpoint split for overlapping refined windows
    for i in range(len(all_refined) - 1):
        if all_refined[i]['end_frame'] > all_refined[i + 1]['start_frame']:
            mid = (all_refined[i]['end_frame'] + all_refined[i + 1]['start_frame']) // 2
            all_refined[i]['end_frame'] = mid
            all_refined[i + 1]['start_frame'] = mid

    shots = []
    for idx, r in enumerate(all_refined, 1):
        sf = r['start_frame']
        ef = r['end_frame']
        shots.append({
            "shot_id": f"{task_id}_shot_{idx}",
            "shot_index": idx,
            "start_frame": sf,
            "end_frame": ef,
            "start_time_sec": round(sf / fps, 2),
            "end_time_sec": round(ef / fps, 2),
            "duration_sec": round((ef - sf) / fps, 2),
            "clip_path": f"processed/{task_id}/shot_{idx}.mp4",
            "thumbnail_path": f"processed/{task_id}/thumb_shot_{idx}.jpg",
            "segmentation_confidence": r.get("confidence", 0.90),
            "parabola_r2": r.get("parabola_r2", 0.0),
            "detected_points": r.get("detected_points", 0),
            "status": "pending"
        })

    contract = {
        "task_id": task_id,
        "video_filename": os.path.basename(video_path),
        "total_video_frames": total_frames,
        "fps": round(fps, 2),
        "shots": shots,
        "shot_count": len(shots)
    }

    # Debug dump if HAWKEYE_SEGMENTATION_DEBUG or debug flag
    if debug or os.environ.get("HAWKEYE_SEGMENTATION_DEBUG") == "1":
        debug_path = os.path.join(os.path.dirname(video_path), "processed", f"{task_id}_segmentation_debug.json")
        try:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({"coarse_windows": coarse, "refined_shots": shots, "contract": contract}, f, indent=2)
        except Exception:
            pass

    return contract


def find_delivery_windows(video_path, model=None, conf_thresh=0.15, pre_pad_sec=1.2, post_pad_sec=1.4, progress_callback=None):
    """
    Two-stage delivery window detector.
    Returns (delivery_windows, was_trimmed), where delivery_windows is [(start_frame, end_frame), ...].
    """
    contract = segment_deliveries(
        video_path, model=model, conf_thresh=conf_thresh, 
        task_id="auto_trim", progress_callback=progress_callback
    )
    shots = contract.get("shots", [])
    if not shots:
        return [(0, contract.get("total_video_frames", 1))], False

    windows = [(s["start_frame"], s["end_frame"]) for s in shots]
    total_frames = contract.get("total_video_frames", 1)
    total_trimmed_len = sum(e - s for (s, e) in windows)

    if len(windows) == 1 and total_trimmed_len >= total_frames * 0.92:
        return [(0, total_frames)], False

    return windows, True


def find_delivery_window(video_path, model=None, conf_thresh=0.15, pre_pad_sec=0.5, post_pad_sec=0.8, progress_callback=None):
    """
    Backward-compatible single delivery window detector.
    """
    windows, was_trimmed = find_delivery_windows(
        video_path, model=model, conf_thresh=conf_thresh, 
        pre_pad_sec=pre_pad_sec, post_pad_sec=post_pad_sec, 
        progress_callback=progress_callback
    )
    if windows and was_trimmed:
        return windows[0][0], windows[0][1], True
    elif windows:
        return windows[0][0], windows[0][1], False
    return 0, 1, False


def extract_shot_clip(input_video_path, output_clip_path, start_frame, end_frame):
    """Extract a precise slice of frames from input video to browser-standard H.264 mp4."""
    os.makedirs(os.path.dirname(os.path.abspath(output_clip_path)), exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    temp_raw = output_clip_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h))

    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    success = convert_to_browser_h264(temp_raw, output_clip_path)
    if success and os.path.exists(output_clip_path):
        try:
            os.remove(temp_raw)
        except Exception:
            pass
        return True
    elif os.path.exists(temp_raw):
        if os.path.exists(output_clip_path):
            try:
                os.remove(output_clip_path)
            except Exception:
                pass
        os.rename(temp_raw, output_clip_path)
        return True
    return False


def capture_shot_thumbnail(input_video_path, output_thumb_path, target_frame):
    """Capture a single representative frame as a high-quality JPEG thumbnail."""
    os.makedirs(os.path.dirname(os.path.abspath(output_thumb_path)), exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(target_frame)))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        cv2.imwrite(output_thumb_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return True
    return False


class KalmanBallTracker:
    """
    2D Kalman Filter for tracking a cricket ball in pixel space with constant velocity & gravity.
    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]
    """
    def __init__(self, init_x, init_y, dt=1.0, gravity=0.5):
        self.dt = dt
        self.gravity = gravity
        # State: [x, y, vx, vy]
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)
        
        # State transition matrix
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Measurement matrix
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Covariance matrices
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.Q = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0]
        ], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * 5.0

    def predict(self):
        """Predict the next state."""
        self.x = self.F @ self.x
        # Add slight gravity acceleration to vertical position/velocity
        self.x[1] += 0.5 * self.gravity * (self.dt ** 2)
        self.x[3] += self.gravity * self.dt
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, z_x, z_y):
        """Update filter state with verified detection measurement."""
        z = np.array([z_x, z_y], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])

    def get_pos(self):
        return int(round(self.x[0])), int(round(self.x[1]))

    def get_velocity(self):
        return float(self.x[2]), float(self.x[3])

    def reflect_bounce(self, restitution=0.55):
        """Reflect vertical velocity upon pitch bounce with restitution damping."""
        self.x[3] = -abs(self.x[3]) * restitution
        self.P[3, 3] = max(self.P[3, 3], 36.0)


class CricketTrajectoryPredictor:
    def __init__(self, model_path=None, conf_thresh=0.20, history_len=20, pred_steps=8, device=None):
        if model_path is None:
            possible_paths = [
                os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'),
                os.path.join('Cricket-Ball-Trajectory-Prediction-master', 'runs', 'detect', 'train5', 'weights', 'best.pt'),
                os.path.join('..', 'runs', 'detect', 'train5', 'weights', 'best.pt'),
                'yolov8s.pt'
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                model_path = 'yolov8s.pt'
        
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        if device is not None:
            try:
                self.model.to(device)
            except Exception:
                pass
        self.conf_thresh = conf_thresh
        self.history_len = history_len
        self.pred_steps = pred_steps

        # Person detector for body-attachment gating
        person_model_paths = [
            'yolov8n.pt',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt'),
            os.path.join('Cricket-Ball-Trajectory-Prediction-master', 'yolov8n.pt'),
            'yolov8s.pt'
        ]
        person_model_path = None
        for p in person_model_paths:
            if os.path.exists(p):
                person_model_path = p
                break
        if person_model_path is None:
            person_model_path = 'yolov8n.pt'
        self.person_model_path = person_model_path
        self._person_model = None

    @property
    def person_model(self):
        if self._person_model is None:
            try:
                self._person_model = YOLO(self.person_model_path)
                if self.device is not None:
                    try:
                        self._person_model.to(self.device)
                    except Exception:
                        pass
            except Exception:
                self._person_model = None
        return self._person_model

    def draw_glowing_path(self, frame, points, color_bgr=(0, 230, 255), core_color=(255, 255, 255), 
                          use_bezier=True, show_nodes=True):
        """Draw a high-visibility Hawkeye glowing path with nodes."""
        if len(points) < 2:
            return

        pts = np.array(points, dtype=np.int32)
        
        # Smooth with Bezier if requested and sufficient points
        if use_bezier and len(points) >= 4:
            smooth_pts = []
            for i in range(0, len(points) - 2, 2):
                sub = [points[i], points[i+1], points[i+2]]
                bez = create_bezier_curve(sub, smoothness=15)
                smooth_pts.extend(bez.tolist())
            if len(smooth_pts) > 0:
                pts = np.array(smooth_pts, dtype=np.int32)

        # Glow layer (thick semi-transparent)
        glow_layer = frame.copy()
        cv2.polylines(glow_layer, [pts], isClosed=False, color=color_bgr, thickness=6, lineType=cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.45, frame, 0.55, 0, frame)

        # Crisp inner core line
        cv2.polylines(frame, [pts], isClosed=False, color=color_bgr, thickness=3, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts], isClosed=False, color=core_color, thickness=1, lineType=cv2.LINE_AA)

        # Node markers on original verified points
        if show_nodes:
            for pt in points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius=3, color=color_bgr, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius=1, color=core_color, thickness=-1, lineType=cv2.LINE_AA)

    def draw_reticle_box(self, frame, x1, y1, x2, y2, label="Ball", color=(0, 220, 255)):
        """Draw modern corner-bracket bounding box with label."""
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = max(int(x2 - x1), 10)
        h = max(int(y2 - y1), 10)
        line_len = max(int(min(w, h) * 0.35), 4)

        # Center target dot
        cv2.circle(frame, (cx, cy), radius=2, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius=5, color=color, thickness=1, lineType=cv2.LINE_AA)

        # Corner brackets
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, 2, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, 2, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, 2, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, 2, cv2.LINE_AA)

        # Tag pill
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        ty = max(y1 - 6, 18)
        cv2.rectangle(frame, (x1, ty - text_size[1] - 3), (x1 + text_size[0] + 6, ty + 3), (20, 25, 35), -1)
        cv2.rectangle(frame, (x1, ty - text_size[1] - 3), (x1 + text_size[0] + 6, ty + 3), color, 1)
        cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def process_video(self, input_video_path, output_video_path=None, show_window=False, 
                      use_bezier=True, persistent_trail=True, auto_trim=True, task_id=None, progress_callback=None):
        """
        Process a video, track cricket ball trajectory with robust outlier rejection,
        smooth Kalman momentum estimation, bounce detection, and broadcast-style annotations.
        Optionally auto-trims to the active delivery window (release to batsman/stumps).
        Generates full annotated video as well as isolated per-shot video clips and thumbnails.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video file not found: {input_video_path}")

        # Resolve task_id if not explicitly provided
        if not task_id and output_video_path:
            base_out = os.path.basename(output_video_path)
            task_id = base_out.replace("pred_", "").replace(".mp4", "")
        if not task_id:
            task_id = "delivery"

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Initial progress notification
        if progress_callback:
            try:
                progress_callback(0, total_frames, 0.0, False, "Auto-detecting delivery action window...")
            except TypeError:
                try:
                    progress_callback(0, total_frames, 0.0, False)
                except Exception:
                    pass

        # Multi-Delivery Auto-trim detection
        start_frame = 0
        end_frame = total_frames
        was_trimmed = False

        if auto_trim:
            windows, was_trimmed = find_delivery_windows(
                input_video_path, self.model, 
                conf_thresh=self.conf_thresh, 
                progress_callback=progress_callback
            )
            if was_trimmed and windows:
                delivery_windows = windows
            else:
                delivery_windows = [(0, total_frames)]
                was_trimmed = False
        else:
            delivery_windows = [(0, total_frames)]
            was_trimmed = False

        effective_total_frames = max(1, sum(w[1] - w[0] for w in delivery_windows))
        total_shots = len(delivery_windows)

        out = None
        temp_raw_path = None
        task_shots_dir = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            temp_raw_path = output_video_path + ".raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
            task_shots_dir = os.path.join(os.path.dirname(os.path.abspath(output_video_path)), task_id)
            os.makedirs(task_shots_dir, exist_ok=True)

        # Tracking state
        kalman = None
        missed_frames = 0
        max_missed_frames = 5  # Bridge gaps up to 5 frames
        max_jump_dist = max(width, height) * 0.25  # Max plausible ball speed per frame

        # --- Physics Validator State ---
        # Minimum confidence for any detection to be considered (raises bar against 53% wristband hits)
        MIN_CONF_DISCOVERY = 0.55        # Needed to START tracking a new ball (no Kalman yet)
        MIN_CONF_TRACKING  = 0.35        # Needed to UPDATE an established Kalman track
        # Once Kalman is established, only accept detections within this pixel radius of prediction
        KALMAN_GATE_RADIUS  = max(width, height) * 0.12   # Tight gate: ~12% of frame
        KALMAN_STRICT_RADIUS = max(width, height) * 0.06  # Ultra-tight if conf < 0.60
        # A real ball must be MOVING – min displacement across last N frames (pixels)
        MIN_VELOCITY_TO_CONFIRM = 5.0   # px/frame – gloves are nearly stationary
        # Stability buffer: require 2 consecutive quality detections before starting Kalman
        pending_candidate = None        # (cx, cy, conf) of tentative first detection
        pending_candidate_count = 0

        # Trajectory stores
        full_trajectory = []     # All tracked (x, y) across current delivery shot
        active_trail = deque(maxlen=self.history_len)
        bounce_events = []
        speed_tracker = CricketSpeedTracker(fps=fps, frame_width=width, frame_height=height)
        current_speed_kmh = 0.0
        current_speed_mph = 0.0
        
        telemetry = {
            "total_frames": effective_total_frames,
            "original_total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "detected_frames": 0,
            "bounce_events": [],
            "speed_summary": None,
            "trajectory_points": [],
            "frame_data": [],
            "total_shots": total_shots,
            "shots_data": [],
            "trim_info": {
                "trimmed": was_trimmed,
                "total_shots": total_shots,
                "shots": [
                    {
                        "shot": idx + 1,
                        "start_frame": w[0],
                        "end_frame": w[1],
                        "start_sec": round(w[0] / fps, 2),
                        "end_sec": round(w[1] / fps, 2),
                        "duration_sec": round((w[1] - w[0]) / fps, 2)
                    } for idx, w in enumerate(delivery_windows)
                ],
                "start_frame": delivery_windows[0][0],
                "end_frame": delivery_windows[-1][1],
                "start_sec": round(delivery_windows[0][0] / fps, 2),
                "end_sec": round(delivery_windows[-1][1] / fps, 2),
                "duration_sec": round(effective_total_frames / fps, 2),
                "saved_time_sec": round((total_frames - effective_total_frames) / fps, 2)
            }
        }

        processed_frames = 0
        prev_time = time.time()
        user_stopped = False
        shots_data = []

        for shot_idx, (w_start, w_end) in enumerate(delivery_windows):
            if user_stopped:
                break

            shot_num = shot_idx + 1
            shot_raw_temp = None
            shot_clip_final = None
            shot_thumb_final = None
            shot_writer = None
            best_thumb_frame = None

            if task_shots_dir:
                shot_raw_temp = os.path.join(task_shots_dir, f"shot_{shot_num}.raw.mp4")
                shot_clip_final = os.path.join(task_shots_dir, f"shot_{shot_num}.mp4")
                shot_thumb_final = os.path.join(task_shots_dir, f"thumb_shot_{shot_num}.jpg")
                fourcc_shot = cv2.VideoWriter_fourcc(*'mp4v')
                shot_writer = cv2.VideoWriter(shot_raw_temp, fourcc_shot, fps, (width, height))

            # -----------------------------------------------------------------
            # Per-Shot State Reset:
            # Whenever a new shot plays, wipe old trajectory ribbon, active trail,
            # Kalman momentum, and previous bounce rings.
            # -----------------------------------------------------------------
            kalman = None
            missed_frames = 0
            pending_candidate = None
            pending_candidate_count = 0
            shot_detected_frames = 0
            shot_angles = []
            full_trajectory.clear()
            active_trail.clear()
            bounce_events.clear()
            speed_tracker.reset_delivery()
            current_speed_kmh = 0.0
            current_speed_mph = 0.0
            last_angle = 0.0
            consecutive_bounces = 0
            last_vy = 0.0

            # Two-Phase Modeling & Outlier Rejection State
            current_phase = 1          # Phase 1: Pre-bounce (release -> bounce). Phase 2: Post-bounce (bounce -> stumps)
            phase_1_points = []
            phase_2_points = []
            bounce_point = None
            stump_hit_point = None
            is_stump_hit = False
            post_bounce_boost_frames = 0
            tracking_stopped_for_shot = False
            consecutive_slow_frames = 0
            bowler_release_zone = None
            frame_buffer = deque()
            pending_bounce = None
            cached_faces = []
            cached_persons = []
            bowler_box = None
            batsman_box = None
            recent_bowler_separations = deque(maxlen=15)
            has_separated_from_bowler = False

            # Seek video directly to the start of this delivery window
            shot_video_start_frame = processed_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, w_start)
            frame_idx = w_start

            while cap.isOpened() and frame_idx < w_end:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_now = time.time()
                instant_fps = 1.0 / max(t_now - prev_time, 1e-5)
                prev_time = t_now

                # Update face mask periodically during discovery
                if (frame_idx % 4 == 0 or kalman is None) and _FACE_CASCADE is not None:
                    cached_faces = detect_faces_fast(frame)

                # Update person detector on 3-frame stride (low-cost, ~16ms/frame on CPU, <1ms GPU)
                if (frame_idx % 3 == 0 or not cached_persons) and self.person_model is not None:
                    try:
                        p_res = self.person_model.predict(frame, classes=[0], conf=0.25, imgsz=320, verbose=False)
                        if len(p_res) > 0 and p_res[0].boxes is not None and len(p_res[0].boxes) > 0:
                            cached_persons = [list(map(float, b)) for b in p_res[0].boxes.xyxy.cpu().numpy()]
                        else:
                            cached_persons = []
                    except Exception:
                        pass

                # Update bowler & batsman classification
                if cached_persons:
                    b_cand, bat_cand = identify_bowler_and_batsman(cached_persons, bowler_release_zone, width, height)
                    if b_cand is not None:
                        bowler_box = b_cand
                    if bat_cand is not None:
                        batsman_box = bat_cand

                # 1. Predict with Kalman filter if active
                predicted_pos = None
                if kalman is not None and not tracking_stopped_for_shot:
                    predicted_pos = kalman.predict()

                # 2. Run YOLO Object Detection
                results = self.model.predict(frame, conf=max(self.conf_thresh * 0.7, 0.12), verbose=False)
                boxes = results[0].boxes if len(results) > 0 else None

                # 3. Filter Candidates & Select True Cricket Ball (Physics Validator)
                best_candidate = None
                highest_score = -1.0
                candidate_is_bounce_rebound = False

                if not tracking_stopped_for_shot and boxes is not None and len(boxes) > 0:
                    box_data = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else [1.0] * len(box_data)

                    for i in range(len(box_data)):
                        x1, y1, x2, y2 = box_data[i]
                        conf = float(confs[i])

                        # --- Gate 1: Shape / Aspect ratio (Cricket ball is roughly 1:1) ---
                        bw = max(x2 - x1, 1)
                        bh = max(y2 - y1, 1)
                        aspect = float(bw) / float(bh)
                        if aspect < 0.35 or aspect > 2.8:
                            continue  # Bats, limbs, long shadows
                        circularity = 1.0 - abs(1.0 - aspect)  # 1.0 = perfect square

                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        # --- Body-Attachment & Role Gating ---
                        dist_to_bowler = get_distance_to_box(cx, cy, bowler_box) if bowler_box is not None else float('inf')
                        dist_to_nearest_person = get_distance_to_nearest_person(cx, cy, cached_persons)

                        # Body adjacency logic:
                        # If bowler_box is known, test distance to bowler.
                        # If bowler_box is None (safe fallback), test distance to nearest person if track hasn't separated.
                        is_near_bowler = (dist_to_bowler < width * 0.12)
                        if bowler_box is None and cached_persons:
                            if not has_separated_from_bowler:
                                is_near_bowler = (dist_to_nearest_person < width * 0.12)
                        elif bowler_box is None and not cached_persons and bowler_release_zone is not None:
                            if not has_separated_from_bowler:
                                dist_from_release = math.hypot(cx - bowler_release_zone[0], cy - bowler_release_zone[1])
                                is_near_bowler = (dist_from_release < width * 0.12)

                        # --- Gate 1A: Absolute Bounding Box Size Ceiling ---
                        # Body-adjacent candidates near bowler get strict ceiling to block wristbands / hands.
                        # Free-flying candidates get relaxed ceiling to comfortably admit close nets ball.
                        if is_near_bowler:
                            max_w_allowed = width * 0.06
                            max_h_allowed = height * 0.07
                            max_area_allowed = width * height * 0.0045
                        else:
                            max_w_allowed = width * 0.11
                            max_h_allowed = height * 0.13
                            max_area_allowed = width * height * 0.013

                        if bw > max_w_allowed or bh > max_h_allowed or (bw * bh) > max_area_allowed:
                            continue

                        # --- Gate 1B: Face Exclusion Mask ---
                        # Reject candidates whose centers fall directly inside a detected human face
                        is_face_hit = False
                        if cached_faces:
                            for (fx, fy, fw, fh) in cached_faces:
                                if (fx + 0.10 * fw <= cx <= fx + 0.90 * fw) and (fy + 0.10 * fh <= cy <= fy + 0.90 * fh):
                                    curr_v = math.hypot(*kalman.get_velocity()) if kalman is not None else 0.0
                                    if kalman is None or curr_v < 12.0:
                                        is_face_hit = True
                                        break
                        if is_face_hit:
                            continue

                        # --- Continuity & Outlier Rejection (Bug A Fix) ---
                        # Once track has >= 6 points and moved down the pitch:
                        if len(full_trajectory) >= 6:
                            last_pt = full_trajectory[-1]
                            if last_pt[1] > height * 0.35:
                                # Candidate jumping > 70 px backwards up the pitch towards bowler
                                if (cy - last_pt[1]) < -70:
                                    continue
                                # Candidate near bowler release zone while current ball is far down pitch
                                if bowler_release_zone is not None:
                                    dist_to_bowler = math.hypot(cx - bowler_release_zone[0], cy - bowler_release_zone[1])
                                    dist_curr_to_bowler = math.hypot(last_pt[0] - bowler_release_zone[0], last_pt[1] - bowler_release_zone[1])
                                    if dist_to_bowler < 65 and dist_curr_to_bowler > 120:
                                        tracking_stopped_for_shot = True
                                        break

                        # --- Gate 2: Spatial Proximity & Confidence (Bug B Fix) ---
                        is_rebound_cand = False
                        if kalman is not None and predicted_pos is not None:
                            pred_x, pred_y = predicted_pos
                            dist = math.hypot(cx - pred_x, cy - pred_y)

                            # Check if candidate matches a reflected bounce trajectory
                            effective_dist = dist
                            if current_phase == 1 and len(full_trajectory) >= 3:
                                vy_est = kalman.x[3]
                                if vy_est > 0.8:
                                    # REBOUND PERMISSION RULES:
                                    # 1. Pitch bounce CANNOT occur on or adjacent to the bowler's body.
                                    #    Track must have separated from bowler (has_separated_from_bowler),
                                    #    and candidate cannot be near bowler (not is_near_bowler).
                                    # 2. Near batsman, rebound is permitted based on parabolic reflection.
                                    rebound_permitted = (not is_near_bowler) and (has_separated_from_bowler or bowler_box is None)

                                    if rebound_permitted:
                                        bounce_y_pred = pred_y - 1.55 * vy_est * kalman.dt
                                        dist_reflected = math.hypot(cx - pred_x, cy - bounce_y_pred)
                                        if dist_reflected < dist and dist_reflected < KALMAN_GATE_RADIUS * 1.8 and cy <= pred_y + 8:
                                            effective_dist = dist_reflected
                                            is_rebound_cand = True

                            base_gate = KALMAN_GATE_RADIUS * 2.2 if post_bounce_boost_frames > 0 else KALMAN_GATE_RADIUS
                            strict_gate = KALMAN_GATE_RADIUS * 1.5 if post_bounce_boost_frames > 0 else KALMAN_STRICT_RADIUS
                            effective_gate = base_gate if conf >= 0.50 else strict_gate

                            if effective_dist > effective_gate:
                                continue

                            min_conf = 0.18 if (post_bounce_boost_frames > 0 or is_rebound_cand) else MIN_CONF_TRACKING
                            if is_near_bowler:
                                min_conf = max(min_conf, 0.45)

                            if conf < min_conf:
                                continue

                            proximity_weight = math.exp(- (effective_dist ** 2) / (2 * (50.0 ** 2)))
                            score = (conf * 0.45) + (proximity_weight * 0.45) + (circularity * 0.10)
                        else:
                            disc_conf = 0.25 if post_bounce_boost_frames > 0 else MIN_CONF_DISCOVERY
                            if is_near_bowler:
                                disc_conf = max(disc_conf, 0.55)
                            if conf < disc_conf:
                                continue
                            score = (conf * 0.75) + (circularity * 0.25)

                        if score > highest_score:
                            highest_score = score
                            best_candidate = (int(x1), int(y1), int(x2), int(y2), cx, cy, conf)
                            candidate_is_bounce_rebound = is_rebound_cand

                # 4. State Update with Stability Buffer (Physics Validator)
                ball_detected = False
                current_centroid = None
                current_bbox = None
                is_estimated = False
                display_conf = 0.0

                if best_candidate is not None:
                    x1, y1, x2, y2, cx, cy, conf = best_candidate

                    # If candidate matches a pitch bounce rebound, reflect Kalman
                    if candidate_is_bounce_rebound and kalman is not None and kalman.x[3] > 0:
                        kalman.reflect_bounce(restitution=0.55)
                        current_phase = 2
                        post_bounce_boost_frames = 8
                        if bounce_point is None and len(full_trajectory) > 0:
                            bounce_point = full_trajectory[-1]
                            pending_bounce = {
                                "shot": shot_idx + 1,
                                "frame": frame_idx,
                                "timestamp": round(frame_idx / fps, 2),
                                "angle": round(last_angle, 1),
                                "speed_kmh": current_speed_kmh,
                                "speed_mph": current_speed_mph,
                                "position": list(bounce_point),
                                "observed_phase2_frames": 0,
                                "phase2_displacements": [],
                                "confirmed": False
                            }

                    if kalman is not None:
                        kx, ky = kalman.update(cx, cy)
                        vx_k, vy_k = kalman.get_velocity()
                        speed = math.hypot(vx_k, vy_k)
                        track_len = len(full_trajectory)
                        if track_len > 4 and speed < MIN_VELOCITY_TO_CONFIRM and conf < 0.70 and current_phase == 1:
                            pass
                        else:
                            ball_detected = True
                            display_conf = conf
                            current_bbox = [x1, y1, x2, y2]
                            current_centroid = (int(kx), int(ky))
                            missed_frames = 0
                            pending_candidate = None
                            pending_candidate_count = 0
                            telemetry["detected_frames"] += 1
                            shot_detected_frames += 1
                            if bowler_release_zone is None:
                                bowler_release_zone = current_centroid

                            # Track bowler separation
                            if bowler_box is not None:
                                sep = get_distance_to_box(cx, cy, bowler_box)
                                recent_bowler_separations.append(sep)
                                if sep >= width * 0.15:
                                    has_separated_from_bowler = True
                            elif bowler_release_zone is not None:
                                sep = math.hypot(cx - bowler_release_zone[0], cy - bowler_release_zone[1])
                                recent_bowler_separations.append(sep)
                                if sep >= width * 0.15:
                                    has_separated_from_bowler = True

                    else:
                        if pending_candidate is not None:
                            px, py, pc = pending_candidate
                            inter_dist = math.hypot(cx - px, cy - py)
                            if inter_dist >= MIN_VELOCITY_TO_CONFIRM and inter_dist < max_jump_dist:
                                kalman = KalmanBallTracker(cx, cy, dt=1.0, gravity=0.4)
                                ball_detected = True
                                display_conf = conf
                                current_bbox = [x1, y1, x2, y2]
                                current_centroid = (int(cx), int(cy))
                                missed_frames = 0
                                pending_candidate = None
                                pending_candidate_count = 0
                                telemetry["detected_frames"] += 1
                                shot_detected_frames += 1
                                if bowler_release_zone is None:
                                    bowler_release_zone = current_centroid

                                # Track bowler separation
                                if bowler_box is not None:
                                    sep = get_distance_to_box(cx, cy, bowler_box)
                                    recent_bowler_separations.append(sep)
                                    if sep >= width * 0.15:
                                        has_separated_from_bowler = True
                                elif bowler_release_zone is not None:
                                    sep = math.hypot(cx - bowler_release_zone[0], cy - bowler_release_zone[1])
                                    recent_bowler_separations.append(sep)
                                    if sep >= width * 0.15:
                                        has_separated_from_bowler = True
                            else:
                                pending_candidate = (cx, cy, conf)
                                pending_candidate_count = 1
                        else:
                            pending_candidate = (cx, cy, conf)
                            pending_candidate_count = 1

                else:
                    pending_candidate = None
                    pending_candidate_count = 0
                # Kalman momentum bridge for missed frames
                if not ball_detected and not is_estimated and not tracking_stopped_for_shot:
                    if kalman is not None and missed_frames < max_missed_frames:
                        missed_frames += 1
                        is_estimated = True
                        kx, ky = kalman.get_pos()
                        if 0 <= kx < width and 0 <= ky < height:
                            current_centroid = (kx, ky)
                            box_rad = 12
                            current_bbox = [max(0, kx - box_rad), max(0, ky - box_rad),
                                            min(width - 1, kx + box_rad), min(height - 1, ky + box_rad)]
                        else:
                            kalman = None
                            tracking_stopped_for_shot = True
                    elif kalman is not None:
                        kalman = None
                        missed_frames = 0

                # 5. Append Trajectory Point & Velocity Calculations
                if post_bounce_boost_frames > 0:
                    post_bounce_boost_frames -= 1

                if current_centroid is not None and not tracking_stopped_for_shot:
                    full_trajectory.append(current_centroid)
                    active_trail.append(current_centroid)
                    current_speed_kmh, current_speed_mph = speed_tracker.update(
                        frame_idx, current_centroid, is_estimated=is_estimated
                    )

                    # Manage Phase 1 and Phase 2 trajectory lists
                    if current_phase == 1:
                        phase_1_points.append(current_centroid)
                    else:
                        if bounce_point and (len(phase_2_points) == 0 or phase_2_points[0] != bounce_point):
                            phase_2_points.insert(0, bounce_point)
                        phase_2_points.append(current_centroid)

                    # Delayed Bounce Confirmation: evaluate subsequent Phase 2 motion
                    if pending_bounce is not None and not pending_bounce["confirmed"]:
                        if len(full_trajectory) >= 2:
                            d_p2 = math.hypot(full_trajectory[-1][0] - full_trajectory[-2][0], full_trajectory[-1][1] - full_trajectory[-2][1])
                            pending_bounce["phase2_displacements"].append(d_p2)
                            pending_bounce["observed_phase2_frames"] += 1

                            if pending_bounce["observed_phase2_frames"] >= 4:
                                p2_disps = pending_bounce["phase2_displacements"]
                                avg_p2_disp = sum(p2_disps) / len(p2_disps)
                                min_p2_disp = min(p2_disps)

                                # Require active ongoing flight rather than dead stop in netting/wall
                                if avg_p2_disp >= 2.0 and min_p2_disp >= 0.8:
                                    pending_bounce["confirmed"] = True
                                    b_info = {
                                        "shot": pending_bounce["shot"],
                                        "frame": pending_bounce["frame"],
                                        "timestamp": pending_bounce["timestamp"],
                                        "angle": pending_bounce["angle"],
                                        "speed_kmh": pending_bounce["speed_kmh"],
                                        "speed_mph": pending_bounce["speed_mph"],
                                        "position": pending_bounce["position"]
                                    }
                                    bounce_events.append(b_info)
                                    telemetry["bounce_events"].append(b_info)
                                    speed_tracker.record_bounce(pending_bounce["frame"])

                                    # Retroactively annotate buffered frames that occurred at or after bounce moment
                                    bx, by = pending_bounce["position"]
                                    b_angle = pending_bounce["angle"]
                                    for buf_item in frame_buffer:
                                        if buf_item["frame_idx"] >= pending_bounce["frame"]:
                                            bf = buf_item["frame"]
                                            cv2.circle(bf, (bx, by), radius=14, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
                                            cv2.circle(bf, (bx, by), radius=7, color=(0, 230, 255), thickness=-1, lineType=cv2.LINE_AA)
                                            cv2.circle(bf, (bx, by), radius=2, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                                            cv2.putText(bf, f"PITCH {b_angle:.0f}°", (bx + 12, by + 4),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
                                            if buf_item["frame_idx"] == pending_bounce["frame"]:
                                                cv2.rectangle(bf, (width - 240, 15), (width - 15, 55), (0, 80, 255), -1)
                                                cv2.putText(bf, "! PITCH BOUNCE !", (width - 225, 42),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                                    print(f"[Physics] Frame {pending_bounce['frame']}: Pitch bounce confirmed ({b_angle:.1f}°).")
                                    pending_bounce = None
                                else:
                                    print(f"[Physics] Frame {pending_bounce['frame']}: Candidate bounce rejected (trajectory halted abruptly in netting/wall, avg disp={avg_p2_disp:.1f}px).")
                                    pending_bounce = None
                                    bounce_point = None

                    # Phase 2 termination check (ground stop or stumps hit)
                    if current_phase == 2 and len(phase_2_points) >= 3:
                        p_last = phase_2_points[-1]
                        p_prev = phase_2_points[-2]
                        disp = math.hypot(p_last[0] - p_prev[0], p_last[1] - p_prev[1])
                        if disp < 2.2:
                            consecutive_slow_frames += 1
                        else:
                            consecutive_slow_frames = 0

                        if consecutive_slow_frames >= 4:
                            tracking_stopped_for_shot = True
                            if not stump_hit_point and p_last[1] > height * 0.40:
                                is_stump_hit = True
                                stump_hit_point = p_last

                # 6. Motion Angle, Future Projection & Bounce Detection
                future_positions = []
                is_bounce = False

                if len(active_trail) >= 2:
                    pts = list(active_trail)
                    x_diff = pts[-1][0] - pts[-2][0]
                    y_diff = pts[-1][1] - pts[-2][1]

                    if x_diff != 0:
                        m1 = y_diff / x_diff
                        if m1 == 1:
                            angle = 90.0
                        elif m1 != 0:
                            angle = 90.0 - angle_between_lines(m1)
                        else:
                            angle = 0.0
                        last_angle = angle
                        shot_angles.append(last_angle)

                    if len(pts) >= 3:
                        prev_ydiff = pts[-2][1] - pts[-3][1]
                        # Rebound: was falling downwards (prev_ydiff > 1.5), now rebounding upward/flattening
                        is_rebound = ((prev_ydiff > 1.5 and y_diff <= 0) or (prev_ydiff > 3.0 and y_diff < prev_ydiff * 0.25)) and pts[-1][1] > height * 0.30
                        if is_rebound and current_phase == 1:
                            consecutive_bounces += 1
                            if consecutive_bounces == 1:
                                current_phase = 2
                                post_bounce_boost_frames = 8
                                if kalman is not None and kalman.x[3] > 0:
                                    kalman.reflect_bounce(restitution=0.55)
                                bounce_pt = (pts[-2][0], pts[-2][1]) if y_diff < 0 else (pts[-1][0], pts[-1][1])
                                if bounce_point is None:
                                    bounce_point = bounce_pt
                                    if phase_1_points and phase_1_points[-1] != bounce_point:
                                        phase_1_points.append(bounce_point)
                                    if not phase_2_points:
                                        phase_2_points.append(bounce_point)
                                pending_bounce = {
                                    "shot": shot_idx + 1,
                                    "frame": frame_idx,
                                    "timestamp": round(frame_idx / fps, 2),
                                    "angle": round(last_angle, 1),
                                    "speed_kmh": current_speed_kmh,
                                    "speed_mph": current_speed_mph,
                                    "position": list(bounce_point) if bounce_point else [pts[-1][0], pts[-1][1]],
                                    "observed_phase2_frames": 0,
                                    "phase2_displacements": [],
                                    "confirmed": False
                                }
                        else:
                            consecutive_bounces = 0

                    vx = x_diff
                    vy = y_diff
                    fx, fy = float(pts[-1][0]), float(pts[-1][1])
                    future_positions.append((int(fx), int(fy)))

                    for step in range(1, self.pred_steps + 1):
                        fx += vx
                        fy += vy + (0.4 * step)
                        if 0 <= fx < width and 0 <= fy < height:
                            future_positions.append((int(fx), int(fy)))
                        else:
                            break

                # 7. Render Broadcast Annotations: Two-Phase Trajectory
                # Phase 1 (Pre-bounce flight): Amber / Gold
                if len(phase_1_points) >= 2:
                    self.draw_glowing_path(frame, phase_1_points, color_bgr=(0, 180, 255), 
                                           core_color=(255, 255, 255), use_bezier=use_bezier, show_nodes=True)
                elif not phase_2_points and len(full_trajectory) >= 2:
                    self.draw_glowing_path(frame, full_trajectory, color_bgr=(0, 180, 255), 
                                           core_color=(255, 255, 255), use_bezier=use_bezier, show_nodes=True)

                # Phase 2 (Post-bounce flight to stumps): Cyan / Electric Emerald
                if len(phase_2_points) >= 2:
                    self.draw_glowing_path(frame, phase_2_points, color_bgr=(255, 220, 0), 
                                           core_color=(255, 255, 255), use_bezier=use_bezier, show_nodes=True)

                for b in bounce_events:
                    bx, by = b["position"]
                    cv2.circle(frame, (bx, by), radius=14, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (bx, by), radius=7, color=(0, 230, 255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (bx, by), radius=2, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.putText(frame, f"PITCH {b['angle']:.0f}°", (bx + 12, by + 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)

                if stump_hit_point is not None:
                    sx, sy = stump_hit_point
                    cv2.circle(frame, (sx, sy), radius=16, color=(0, 50, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), radius=8, color=(0, 180, 255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), radius=3, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.putText(frame, "WICKET HIT", (sx + 14, sy + 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 2, cv2.LINE_AA)

                if len(future_positions) >= 2:
                    for i in range(1, len(future_positions)):
                        cv2.line(frame, future_positions[i - 1], future_positions[i], (0, 255, 128), 2, cv2.LINE_AA)
                        cv2.circle(frame, future_positions[i], radius=3, color=(0, 255, 200), thickness=-1, lineType=cv2.LINE_AA)

                if current_bbox is not None and current_centroid is not None:
                    x1, y1, x2, y2 = current_bbox
                    tag = f"Ball {display_conf*100:.0f}%" if not is_estimated else "Track (EST)"
                    tag_color = (0, 220, 255) if not is_estimated else (0, 165, 255)
                    self.draw_reticle_box(frame, x1, y1, x2, y2, label=tag, color=tag_color)

                hud_bg = frame.copy()
                hud_w = 340
                hud_h = 76
                cv2.rectangle(hud_bg, (12, 12), (hud_w, hud_h), (15, 18, 26), -1)
                cv2.addWeighted(hud_bg, 0.78, frame, 0.22, 0, frame)
                cv2.rectangle(frame, (12, 12), (hud_w, hud_h), (55, 70, 95), 1)

                processed_frames += 1

                shot_str = f"[SHOT {shot_idx + 1}/{total_shots}] " if total_shots > 1 else ""
                speed_val_str = f"{current_speed_kmh:.1f} km/h | {current_speed_mph:.1f} mph" if current_speed_kmh > 0 else "-- km/h"
                speed_str = f"{shot_str}SPEED: {speed_val_str}"
                speed_color = (0, 255, 128) if current_speed_kmh > 0 else (160, 160, 160)
                cv2.putText(frame, speed_str, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44 if total_shots > 1 else 0.48, speed_color, 2, cv2.LINE_AA)

                if len(bounce_events) > 0:
                    angle_str = f"BOUNCE ANGLE: {bounce_events[-1]['angle']:.1f}°"
                else:
                    angle_str = f"ANGLE: {last_angle:.1f}°"
                cv2.putText(frame, angle_str, (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2, cv2.LINE_AA)

                # Flash pitch bounce banner ONLY on confirmed bounce frame
                is_bounce_confirmed = any(b.get("frame") == frame_idx for b in bounce_events)
                if is_bounce_confirmed:
                    cv2.rectangle(frame, (width - 240, 15), (width - 15, 55), (0, 80, 255), -1)
                    cv2.putText(frame, "! PITCH BOUNCE !", (width - 225, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

                # Capture candidate thumbnail frame
                if is_bounce_confirmed or best_thumb_frame is None or (ball_detected and display_conf > 0.65):
                    best_thumb_frame = frame.copy()

                frame_entry = {
                    "shot": shot_idx + 1,
                    "frame": frame_idx,
                    "relative_frame": processed_frames,
                    "phase": current_phase,
                    "detected": ball_detected,
                    "is_estimated": is_estimated,
                    "centroid": current_centroid,
                    "bbox": current_bbox,
                    "speed_kmh": current_speed_kmh,
                    "speed_mph": current_speed_mph,
                    "angle": round(last_angle, 2),
                    "is_bounce": is_bounce_confirmed,
                    "is_stump_hit": is_stump_hit,
                    "predicted_points": future_positions
                }
                telemetry["frame_data"].append(frame_entry)
                if current_centroid:
                    telemetry["trajectory_points"].append({
                        "shot": shot_idx + 1,
                        "frame": frame_idx,
                        "relative_frame": processed_frames,
                        "phase": current_phase,
                        "x": current_centroid[0],
                        "y": current_centroid[1],
                        "is_estimated": is_estimated,
                        "is_bounce": is_bounce_confirmed,
                        "is_stump_hit": is_stump_hit
                    })

                # Buffer frame for delayed confirmation before writing to disk
                frame_buffer.append({"frame": frame, "frame_idx": frame_idx})
                if len(frame_buffer) > 6:
                    item_to_write = frame_buffer.popleft()
                    if out is not None:
                        out.write(item_to_write["frame"])
                    if shot_writer is not None:
                        shot_writer.write(item_to_write["frame"])

                # Live preview window if requested
                if show_window:
                    resized = cv2.resize(frame, (1000, 600))
                    cv2.imshow("Cricket Ball Trajectory Prediction", resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        user_stopped = True
                        break
                    elif key == ord(' '):
                        while True:
                            k = cv2.waitKey(30) & 0xFF
                            if k == ord('q'):
                                user_stopped = True
                                break
                            elif k == ord(' '):
                                break

                if progress_callback:
                    step_msg = f"Tracking delivery {shot_idx + 1} of {total_shots}..." if total_shots > 1 else "Tracking ball trajectory & kinematics..."
                    try:
                        progress_callback(processed_frames, effective_total_frames, last_angle, ball_detected or is_estimated, step_msg)
                    except TypeError:
                        try:
                            progress_callback(processed_frames, effective_total_frames, last_angle, ball_detected or is_estimated)
                        except Exception:
                            pass

            # Flush any remaining buffered frames to disk
            while frame_buffer:
                item_to_write = frame_buffer.popleft()
                if out is not None:
                    out.write(item_to_write["frame"])
                if shot_writer is not None:
                    shot_writer.write(item_to_write["frame"])

            shot_video_end_frame = processed_frames
            shot_video_start_sec = round(shot_video_start_frame / fps, 2)
            shot_video_end_sec = round(shot_video_end_frame / fps, 2)

            # Finalize per-shot video clip
            if shot_writer is not None:
                shot_writer.release()
                if os.path.exists(shot_raw_temp):
                    success = convert_to_browser_h264(shot_raw_temp, shot_clip_final)
                    if success and os.path.exists(shot_clip_final):
                        try:
                            os.remove(shot_raw_temp)
                        except Exception:
                            pass
                    else:
                        if os.path.exists(shot_clip_final):
                            try:
                                os.remove(shot_clip_final)
                            except Exception:
                                pass
                        os.rename(shot_raw_temp, shot_clip_final)

            # Save per-shot thumbnail
            if best_thumb_frame is not None and shot_thumb_final:
                try:
                    cv2.imwrite(shot_thumb_final, best_thumb_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                except Exception:
                    pass

            # Record per-shot kinematics summary
            curr_shot_speed = speed_tracker.get_summary(include_history=False)
            shot_total_frames = max(1, w_end - w_start)
            shot_info = {
                "shot_id": f"{task_id}_shot_{shot_num}",
                "shot_index": shot_num,
                "shot": shot_num,
                "start_frame": w_start,
                "end_frame": w_end,
                "start_sec": round(w_start / fps, 2),
                "end_sec": round(w_end / fps, 2),
                "video_start_frame": shot_video_start_frame,
                "video_end_frame": shot_video_end_frame,
                "video_start_sec": shot_video_start_sec,
                "video_end_sec": shot_video_end_sec,
                "duration_sec": round((w_end - w_start) / fps, 2),
                "total_frames": shot_total_frames,
                "detected_frames": shot_detected_frames,
                "tracking_rate": round((shot_detected_frames / shot_total_frames) * 100, 1),
                "release_speed_kmh": curr_shot_speed.get("release_speed_kmh", 0) or 0.0,
                "release_speed_mph": curr_shot_speed.get("release_speed_mph", 0) or 0.0,
                "avg_speed_kmh": curr_shot_speed.get("avg_speed_kmh", 0) or 0.0,
                "avg_speed_mph": curr_shot_speed.get("avg_speed_mph", 0) or 0.0,
                "max_speed_kmh": curr_shot_speed.get("max_speed_kmh", 0) or 0.0,
                "max_speed_mph": curr_shot_speed.get("max_speed_mph", 0) or 0.0,
                "bounces": list(bounce_events),
                "bounce_count": len(bounce_events),
                "avg_angle": round(sum(shot_angles) / len(shot_angles), 1) if shot_angles else 0.0,
                "phase_1_trajectory": list(phase_1_points),
                "phase_2_trajectory": list(phase_2_points),
                "bounce_point": list(bounce_point) if bounce_point else None,
                "stump_hit_point": list(stump_hit_point) if stump_hit_point else None,
                "is_stump_hit": is_stump_hit,
                "clip_path": f"processed/{task_id}/shot_{shot_num}.mp4" if task_shots_dir else "",
                "thumbnail_path": f"processed/{task_id}/thumb_shot_{shot_num}.jpg" if task_shots_dir else "",
                "video_url": f"/videos/processed/{task_id}/shot_{shot_num}.mp4" if task_shots_dir else "",
                "thumbnail_url": f"/videos/processed/{task_id}/thumb_shot_{shot_num}.jpg" if task_shots_dir else "",
                "status": "processed"
            }
            shots_data.append(shot_info)

        cap.release()
        if out is not None:
            out.release()
            if temp_raw_path and os.path.exists(temp_raw_path):
                success = convert_to_browser_h264(temp_raw_path, output_video_path)
                if success and os.path.exists(output_video_path):
                    try:
                        os.remove(temp_raw_path)
                    except Exception:
                        pass
                else:
                    if os.path.exists(output_video_path):
                        try:
                            os.remove(output_video_path)
                        except Exception:
                            pass
                    os.rename(temp_raw_path, output_video_path)

        if show_window:
            cv2.destroyAllWindows()

        telemetry["task_id"] = task_id
        telemetry["shots"] = shots_data
        telemetry["shots_data"] = shots_data
        telemetry["shot_count"] = len(shots_data)
        telemetry["speed_summary"] = speed_tracker.get_summary()
        telemetry["trajectory"] = {
            "pre_bounce": [p for p in telemetry["trajectory_points"] if p.get("phase") == 1],
            "post_bounce": [p for p in telemetry["trajectory_points"] if p.get("phase") == 2],
            "bounce_point": telemetry["bounce_events"][0]["position"] if telemetry["bounce_events"] else None,
            "stump_hit": [s["stump_hit_point"] for s in shots_data if s.get("stump_hit_point")] or None
        }
        return telemetry
