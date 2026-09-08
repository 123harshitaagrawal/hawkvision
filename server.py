import os
import uuid
import threading
import json
import traceback
import datetime
import mimetypes
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from trajectory_tracker import CricketTrajectoryPredictor

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
PROCESSED_DIR = os.path.join(BASE_DIR, 'videos', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

HISTORY_INDEX_PATH = os.path.join(PROCESSED_DIR, 'history.json')

# In-memory task tracker
tasks = {}

# Minimum validity floor constants for bowling deliveries
MIN_VALID_RELEASE_KMH = 20.0
MIN_TRACKING_RATE_PCT = 15.0
MIN_DETECTED_FRAMES = 4

def is_valid_delivery_shot(s):
    """Determine if a segmented shot meets minimum physical validity criteria for a cricket delivery."""
    if not s or not isinstance(s, dict):
        return False
    rel = float(s.get("release_speed_kmh", 0) or 0)
    avg_s = float(s.get("avg_speed_kmh", 0) or 0)
    max_s = float(s.get("max_speed_kmh", 0) or 0)
    det_f = int(s.get("detected_frames", 0) or 0)
    rate = float(s.get("tracking_rate", 0) or 0)
    eff_speed = max(rel, avg_s, max_s)
    # Must achieve at least 20 km/h and have at least minimal verified tracked frames
    return eff_speed >= MIN_VALID_RELEASE_KMH and (rate >= MIN_TRACKING_RATE_PCT or det_f >= MIN_DETECTED_FRAMES)

# ─────────────────────────────────────────────
#  HISTORY MANAGER
# ─────────────────────────────────────────────

def _load_history_index():
    """Load the history index from disk, returning list of records."""
    if os.path.exists(HISTORY_INDEX_PATH):
        try:
            with open(HISTORY_INDEX_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def _save_history_index(records):
    """Persist the history index list to disk."""
    try:
        with open(HISTORY_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, default=str)
    except Exception as e:
        print(f'[History] Failed to save index: {e}')


def _save_task_telemetry(task_id, task_data):
    """Save full task telemetry to a per-task JSON file."""
    telemetry_path = os.path.join(PROCESSED_DIR, f'pred_{task_id}.json')
    try:
        with open(telemetry_path, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2, default=str)
    except Exception as e:
        print(f'[History] Failed to save telemetry for {task_id}: {e}')


def extract_or_build_shots_data(tel):
    """
    Extract shots_data from telemetry, or build it dynamically
    for historical tasks from trim_info, frame_data, and bounce_events.
    """
    if not tel or not isinstance(tel, dict):
        return []

    fps = float(tel.get("fps", 30.0) or 30.0)
    task_id = tel.get("task_id", "")
    existing = tel.get("shots_data")
    if existing and isinstance(existing, list) and len(existing) > 0:
        video_acc = 0.0
        for s in existing:
            dur = float(s.get("duration_sec") or round(s.get("total_frames", 30) / max(fps, 1.0), 2))
            if s.get("video_start_sec") is None:
                s["video_start_sec"] = round(video_acc, 2)
            if s.get("video_end_sec") is None:
                s["video_end_sec"] = round(video_acc + dur, 2)
            video_acc += dur
            shot_num = s.get("shot", 1)
            s["shot_index"] = shot_num
            if not s.get("shot_id"):
                s["shot_id"] = f"{task_id}_shot_{shot_num}" if task_id else f"shot_{shot_num}"
            if not s.get("video_url") and task_id:
                s["video_url"] = f"/videos/processed/{task_id}/shot_{shot_num}.mp4"
            if not s.get("thumbnail_url") and task_id:
                s["thumbnail_url"] = f"/videos/processed/{task_id}/thumb_shot_{shot_num}.jpg"
            if "is_best" not in s:
                s["is_best"] = False
        return existing

    trim_info = tel.get("trim_info", {}) or {}
    shots = trim_info.get("shots", []) or []
    speed_summary = tel.get("speed_summary", {}) or {}
    all_deliveries = speed_summary.get("all_deliveries", []) or []
    bounce_events = tel.get("bounce_events", []) or []
    frame_data = tel.get("frame_data", []) or []
    fps = float(tel.get("fps", 30.0) or 30.0)
    total_frames = tel.get("total_frames", 0)

    # If no trim_info shots listed, default to 1 shot covering all frames
    if not shots:
        shots = [{
            "shot": 1,
            "start_frame": 0,
            "end_frame": total_frames,
            "start_sec": 0.0,
            "end_sec": round(total_frames / max(fps, 1.0), 2),
            "duration_sec": round(total_frames / max(fps, 1.0), 2)
        }]

    total_shots = len(shots)
    built_shots = []
    video_acc = 0.0

    for s_idx, s in enumerate(shots):
        shot_num = s.get("shot", s_idx + 1)
        w_start = s.get("start_frame", 0)
        w_end = s.get("end_frame", total_frames)
        shot_frames = max(1, w_end - w_start)
        dur_sec = s.get("duration_sec", round(shot_frames / max(fps, 1.0), 2))
        v_start = s.get("video_start_sec") if ("video_start_sec" in s and s["video_start_sec"] is not None) else round(video_acc, 2)
        v_end = s.get("video_end_sec") if ("video_end_sec" in s and s["video_end_sec"] is not None) else round(video_acc + dur_sec, 2)
        video_acc += dur_sec

        # Kinematics mapping:
        # all_deliveries holds summaries of shots 0..total_shots-2, speed_summary holds shot total_shots-1
        if s_idx < len(all_deliveries):
            shot_spd = all_deliveries[s_idx] or {}
        else:
            shot_spd = speed_summary or {}

        # Bounce events belonging to this shot window
        shot_bounces = [
            b for b in bounce_events
            if b.get("shot") == shot_num or (w_start <= b.get("frame", -1) <= w_end)
        ]
        for b in shot_bounces:
            b["shot"] = shot_num

        # Frame data for this shot
        shot_fd = [f for f in frame_data if f.get("shot") == shot_num or (w_start <= f.get("frame", -1) <= w_end)]
        det_count = sum(1 for f in shot_fd if f.get("detected") or f.get("centroid"))
        angles = [f.get("angle") for f in shot_fd if f.get("angle") is not None and f.get("angle") > 0]
        avg_ang = round(sum(angles) / len(angles), 1) if angles else 0.0

        rel_k = float(shot_spd.get("release_speed_kmh", 0) or 0.0)
        avg_k = float(shot_spd.get("avg_speed_kmh", 0) or 0.0)
        max_k = float(shot_spd.get("max_speed_kmh", 0) or 0.0)

        # Fallback for release speed if 0
        if rel_k <= 0 and avg_k > 0:
            rel_k = round(avg_k * 0.95, 1)

        shot_id = f"{task_id}_shot_{shot_num}" if task_id else f"shot_{shot_num}"
        v_url = f"/videos/processed/{task_id}/shot_{shot_num}.mp4" if task_id else ""
        t_url = f"/videos/processed/{task_id}/thumb_shot_{shot_num}.jpg" if task_id else ""

        existing_s = next((x for x in (tel.get("shots_data") or []) if x.get("shot") == shot_num), {})

        built_shots.append({
            "shot_id": shot_id,
            "shot_index": shot_num,
            "shot": shot_num,
            "start_frame": w_start,
            "end_frame": w_end,
            "start_sec": s.get("start_sec", round(w_start / max(fps, 1.0), 2)),
            "end_sec": s.get("end_sec", round(w_end / max(fps, 1.0), 2)),
            "video_start_sec": v_start,
            "video_end_sec": v_end,
            "video_start_frame": s.get("video_start_frame", int(round(v_start * fps))),
            "video_end_frame": s.get("video_end_frame", int(round(v_end * fps))),
            "duration_sec": dur_sec,
            "total_frames": shot_frames,
            "detected_frames": det_count if det_count > 0 else max(1, tel.get("detected_frames", 0) // total_shots),
            "tracking_rate": round((det_count / shot_frames) * 100, 1) if shot_frames > 0 and det_count > 0 else round(tel.get("detected_frames", 0) / max(total_frames, 1) * 100, 1),
            "release_speed_kmh": round(rel_k, 1),
            "release_speed_mph": round(rel_k * 0.621371, 1),
            "avg_speed_kmh": round(avg_k, 1),
            "avg_speed_mph": round(avg_k * 0.621371, 1),
            "max_speed_kmh": round(max_k, 1),
            "max_speed_mph": round(max_k * 0.621371, 1),
            "bounces": shot_bounces,
            "bounce_count": len(shot_bounces),
            "avg_angle": avg_ang,
            "phase_1_trajectory": existing_s.get("phase_1_trajectory", []),
            "phase_2_trajectory": existing_s.get("phase_2_trajectory", []),
            "bounce_point": existing_s.get("bounce_point", shot_bounces[0]["position"] if shot_bounces else None),
            "stump_hit_point": existing_s.get("stump_hit_point", None),
            "is_stump_hit": existing_s.get("is_stump_hit", False),
            "video_url": v_url,
            "thumbnail_url": t_url,
            "is_best": False
        })

    return built_shots


def _load_task_telemetry(task_id):
    """Load full task telemetry from per-task JSON file."""
    telemetry_path = os.path.join(PROCESSED_DIR, f'pred_{task_id}.json')
    if os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data and isinstance(data, dict):
                    tel = data.get("telemetry")
                    if tel and isinstance(tel, dict):
                        if "shots_data" not in tel or not tel["shots_data"] or any(s.get("video_start_sec") is None for s in tel["shots_data"]):
                            tel["shots_data"] = extract_or_build_shots_data(tel)
                return data
        except Exception:
            pass
    return None


def _record_to_index(task_id, task_data):
    """Build a lightweight summary record for the history index."""
    tel = task_data.get('telemetry') or {}
    spd = tel.get('speed_summary') or {}
    trim = tel.get('trim_info') or {}
    shots = tel.get('shots_data') or []
    tot_shots = len(shots) if shots else trim.get('total_shots', 1)
    shots_summary = []
    for s in shots:
        shots_summary.append({
            "shot": s.get("shot"),
            "shot_id": s.get("shot_id"),
            "duration_sec": s.get("duration_sec"),
            "release_speed_kmh": s.get("release_speed_kmh"),
            "max_speed_kmh": s.get("max_speed_kmh"),
            "bounce_count": s.get("bounce_count"),
            "thumbnail_url": s.get("thumbnail_url"),
            "video_url": s.get("video_url"),
            "is_best": s.get("is_best", False),
            "is_valid_delivery": s.get("is_valid_delivery", False)
        })
    valid_cnt = sum(1 for s in shots if is_valid_delivery_shot(s))
    return {
        'task_id': task_id,
        'timestamp': task_data.get('timestamp', datetime.datetime.now().isoformat()),
        'input_filename': task_data.get('input_filename', ''),
        'output_filename': task_data.get('output_filename', f'pred_{task_id}.mp4'),
        'status': task_data.get('status', 'completed'),
        'total_frames': tel.get('total_frames', 0),
        'detected_frames': tel.get('detected_frames', 0),
        'fps': tel.get('fps', 30.0),
        'duration_sec': round(tel.get('total_frames', 0) / max(tel.get('fps', 30.0), 1), 2),
        'release_speed_kmh': spd.get('release_speed_kmh', 0),
        'avg_speed_kmh': spd.get('avg_speed_kmh', 0),
        'max_speed_kmh': spd.get('max_speed_kmh', 0),
        'bounce_count': len(tel.get('bounce_events', [])),
        'total_shots': tot_shots,
        'shot_count': tot_shots,
        'valid_shot_count': valid_cnt,
        'has_valid_delivery': task_data.get('has_valid_delivery', valid_cnt > 0),
        'best_shot': task_data.get('best_shot'),
        'shots': shots_summary,
        'has_insights': bool(task_data.get('insights')),
    }


def _add_to_history(task_id, task_data):
    """Add or update a task record in history.json."""
    records = _load_history_index()
    # Remove existing entry for this task_id if present
    records = [r for r in records if r.get('task_id') != task_id]
    new_record = _record_to_index(task_id, task_data)
    records.insert(0, new_record)  # newest first
    _save_history_index(records)
    _save_task_telemetry(task_id, task_data)


def _scan_and_rebuild_history():
    """
    On startup: scan videos/processed/ for pred_*.mp4 files that already have
    a matching pred_*.json telemetry file and index any not already present.
    """
    records = _load_history_index()
    indexed_ids = {r['task_id'] for r in records}
    added = 0
    for fname in os.listdir(PROCESSED_DIR):
        if not (fname.startswith('pred_') and fname.endswith('.mp4') and not fname.endswith('.raw.mp4')):
            continue
        task_id = fname[len('pred_'):-len('.mp4')]
        if task_id in indexed_ids:
            continue
        # Try to load existing telemetry JSON
        tel_data = _load_task_telemetry(task_id)
        if tel_data:
            record = _record_to_index(task_id, tel_data)
        else:
            # Build a minimal record from file metadata
            fpath = os.path.join(PROCESSED_DIR, fname)
            mtime = os.path.getmtime(fpath)
            record = {
                'task_id': task_id,
                'timestamp': datetime.datetime.fromtimestamp(mtime).isoformat(),
                'input_filename': 'unknown',
                'output_filename': fname,
                'status': 'completed',
                'total_frames': 0,
                'detected_frames': 0,
                'fps': 30.0,
                'duration_sec': 0.0,
                'release_speed_kmh': 0,
                'avg_speed_kmh': 0,
                'max_speed_kmh': 0,
                'bounce_count': 0,
                'total_shots': 1,
                'has_insights': False,
            }
        records.append(record)
        added += 1

    if added:
        # Sort by timestamp descending
        records.sort(key=lambda r: r.get('timestamp', ''), reverse=True)
        _save_history_index(records)
        print(f'[History] Indexed {added} existing processed video(s).')


# Run startup scanner
_scan_and_rebuild_history()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/videos', methods=['GET'])
def get_videos():
    sample_videos = []
    if os.path.exists(VIDEOS_DIR):
        for f in os.listdir(VIDEOS_DIR):
            if f.endswith('.mp4') and not f.startswith('output_') and not os.path.isdir(os.path.join(VIDEOS_DIR, f)):
                sample_videos.append({
                    "id": f,
                    "name": f,
                    "size_mb": round(os.path.getsize(os.path.join(VIDEOS_DIR, f)) / (1024 * 1024), 2)
                })
    return jsonify({
        "sample_videos": sample_videos,
        "models": [
            {"id": "runs/detect/train5/weights/best.pt", "name": "Custom Cricket Ball YOLOv8 (best.pt)"},
            {"id": "yolov8s.pt", "name": "YOLOv8 Small (Pretrained)"},
            {"id": "yolov8m.pt", "name": "YOLOv8 Medium (Pretrained)"},
            {"id": "yolov8l.pt", "name": "YOLOv8 Large (Pretrained)"},
            {"id": "yolov8n.pt", "name": "YOLOv8 Nano (Pretrained)"}
        ]
    })

@app.route('/api/predict', methods=['POST'])
def start_prediction():
    video_source = request.form.get('video_source', 'preset')
    model_choice = request.form.get('model', 'runs/detect/train5/weights/best.pt')
    conf_thresh = float(request.form.get('conf', 0.25))
    history_len = int(request.form.get('history', 15))
    pred_steps = int(request.form.get('future', 6))
    use_bezier = request.form.get('bezier', 'true').lower() == 'true'
    persistent_trail = request.form.get('persistent', 'true').lower() == 'true'
    auto_trim = request.form.get('auto_trim', 'true').lower() == 'true'

    task_id = str(uuid.uuid4())[:8]

    # Resolve input video
    if video_source == 'upload' and 'file' in request.files:
        uploaded_file = request.files['file']
        filename = f"upload_{task_id}_{uploaded_file.filename}"
        input_path = os.path.join(VIDEOS_DIR, filename)
        uploaded_file.save(input_path)
    else:
        preset_name = request.form.get('preset_video', 'test1.mp4')
        input_path = os.path.join(VIDEOS_DIR, preset_name)

    if not os.path.exists(input_path):
        return jsonify({"error": f"Input video not found: {input_path}"}), 404

    output_filename = f"pred_{task_id}.mp4"
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    # Initialize task state
    tasks[task_id] = {
        "status": "processing",
        "stage": "segmenting",
        "progress": 0,
        "current_frame": 0,
        "total_frames": 1,
        "angle": 0.0,
        "ball_detected": False,
        "shot_count": 0,
        "current_shot_index": 0,
        "completed_shots": 0,
        "best_shot": 1,
        "step_name": "Initializing model & scanning video...",
        "output_filename": output_filename,
        "input_filename": os.path.basename(input_path),
        "telemetry": None,
        "error": None
    }

    # Run in background thread
    def run_worker():
        try:
            # Resolve model path
            resolved_model = os.path.join(BASE_DIR, model_choice) if not os.path.isabs(model_choice) else model_choice
            if not os.path.exists(resolved_model):
                resolved_model = os.path.join(BASE_DIR, 'runs', 'detect', 'train5', 'weights', 'best.pt')

            predictor = CricketTrajectoryPredictor(
                model_path=resolved_model,
                conf_thresh=conf_thresh,
                history_len=history_len,
                pred_steps=pred_steps
            )

            def update_progress(frame, total, angle, detected, step_name="Predicting ball trajectory..."):
                if task_id in tasks:
                    tasks[task_id]["current_frame"] = frame
                    tasks[task_id]["total_frames"] = total
                    tasks[task_id]["progress"] = max(1, min(99, int((frame / max(total, 1)) * 100)))
                    tasks[task_id]["angle"] = round(angle, 1)
                    tasks[task_id]["ball_detected"] = detected
                    tasks[task_id]["step_name"] = step_name
                    if "Pass A" in step_name or "Pass B" in step_name or "Scanning" in step_name or "Auto-detecting" in step_name:
                        tasks[task_id]["stage"] = "segmenting"
                    elif "Tracking" in step_name or "Predicting" in step_name:
                        tasks[task_id]["stage"] = "tracking"

            telemetry = predictor.process_video(
                input_video_path=input_path,
                output_video_path=output_path,
                show_window=False,
                use_bezier=use_bezier,
                persistent_trail=persistent_trail,
                auto_trim=auto_trim,
                progress_callback=update_progress,
                task_id=task_id
            )

            shots_data = extract_or_build_shots_data(telemetry)
            valid_shots = [s for s in shots_data if is_valid_delivery_shot(s)]
            best_shot_idx = None
            has_valid_delivery = False

            if valid_shots:
                best_item = max(
                    valid_shots,
                    key=lambda s: (float(s.get("release_speed_kmh", 0) or 0) + float(s.get("max_speed_kmh", 0) or 0))
                )
                best_shot_idx = best_item.get("shot", 1)
                has_valid_delivery = True

            for s in shots_data:
                s["is_best"] = (best_shot_idx is not None and s.get("shot") == best_shot_idx)
                s["is_valid_delivery"] = is_valid_delivery_shot(s)

            telemetry["shots_data"] = shots_data
            telemetry["shots"] = shots_data
            telemetry["shot_count"] = len(shots_data)
            telemetry["valid_shot_count"] = len(valid_shots)
            telemetry["has_valid_delivery"] = has_valid_delivery
            telemetry["best_shot"] = best_shot_idx

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["stage"] = "completed"
            tasks[task_id]["progress"] = 100
            tasks[task_id]["shot_count"] = len(shots_data)
            tasks[task_id]["valid_shot_count"] = len(valid_shots)
            tasks[task_id]["has_valid_delivery"] = has_valid_delivery
            tasks[task_id]["completed_shots"] = len(shots_data)
            tasks[task_id]["best_shot"] = best_shot_idx
            tasks[task_id]["current_frame"] = telemetry.get("total_frames", tasks[task_id]["total_frames"])
            tasks[task_id]["total_frames"] = telemetry.get("total_frames", tasks[task_id]["total_frames"])
            tasks[task_id]["step_name"] = "Analysis Complete!"
            tasks[task_id]["telemetry"] = telemetry
            tasks[task_id]["timestamp"] = datetime.datetime.now().isoformat()

            # Persist to history
            _add_to_history(task_id, tasks[task_id])

        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["stage"] = "failed"
            tasks[task_id]["error"] = str(e)

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "processing"})

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id not in tasks:
        loaded = _load_task_telemetry(task_id)
        if loaded:
            tasks[task_id] = loaded
            return jsonify(loaded)
        return jsonify({"error": "Task not found"}), 404
    t = tasks[task_id]
    if t.get("telemetry"):
        tel = t["telemetry"]
        if "shots_data" not in tel or not tel["shots_data"] or any(s.get("video_start_sec") is None for s in tel["shots_data"]):
            tel["shots_data"] = extract_or_build_shots_data(tel)
    try:
        return jsonify(tasks[task_id])
    except Exception as e:
        print(f"Serialization warning on task {task_id}: {e}")
        t = tasks[task_id]
        safe_copy = {
            "task_id": task_id,
            "status": t.get("status", "completed"),
            "progress": t.get("progress", 100),
            "step_name": t.get("step_name", "Analysis Complete!"),
            "output_filename": t.get("output_filename"),
            "total_frames": t.get("total_frames", 1),
            "current_frame": t.get("current_frame", 1),
            "angle": t.get("angle", 0.0),
            "ball_detected": t.get("ball_detected", False),
            "telemetry": {
                "total_frames": t.get("telemetry", {}).get("total_frames", 1) if t.get("telemetry") else 1,
                "fps": t.get("telemetry", {}).get("fps", 30.0) if t.get("telemetry") else 30.0,
                "speed_summary": {
                    "release_speed_kmh": 0.0,
                    "avg_speed_kmh": 0.0,
                    "max_speed_kmh": 0.0
                },
                "bounce_events": [],
                "frame_data": [],
                "trajectory_points": []
            }
        }
        return jsonify(safe_copy)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    if filename.startswith('processed/'):
        actual_file = filename.replace('processed/', '')
        mime, _ = mimetypes.guess_type(actual_file)
        if not mime:
            mime = 'video/mp4' if actual_file.endswith('.mp4') else 'application/octet-stream'
        return send_from_directory(PROCESSED_DIR, actual_file, mimetype=mime, conditional=True)
    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        mime = 'video/mp4' if filename.endswith('.mp4') else 'application/octet-stream'
    return send_from_directory(VIDEOS_DIR, filename, mimetype=mime, conditional=True)


@app.route('/api/task/<task_id>/shots', methods=['GET'])
def get_task_shots(task_id):
    """Return summary metadata for all shots in a task."""
    if task_id not in tasks:
        loaded = _load_task_telemetry(task_id)
        if loaded:
            tasks[task_id] = loaded
        else:
            return jsonify({"error": "Task not found"}), 404
    t = tasks[task_id]
    tel = t.get("telemetry") or {}
    shots = extract_or_build_shots_data(tel)
    
    # Determine best shot using validity filter
    valid_shots = [s for s in shots if is_valid_delivery_shot(s)]
    best_shot = None
    if t.get("insights") and isinstance(t["insights"], dict) and t["insights"].get("best_shot"):
        best_shot = t["insights"].get("best_shot")
    elif valid_shots:
        best_item = max(
            valid_shots,
            key=lambda s: (float(s.get("release_speed_kmh", 0) or 0) + float(s.get("max_speed_kmh", 0) or 0))
        )
        best_shot = best_item.get("shot", 1)

    for s in shots:
        s["is_best"] = (best_shot is not None and s.get("shot") == best_shot)
        s["is_valid_delivery"] = is_valid_delivery_shot(s)

    return jsonify({
        "task_id": task_id,
        "shot_count": len(shots),
        "valid_shot_count": len(valid_shots),
        "has_valid_delivery": bool(valid_shots),
        "best_shot": best_shot,
        "shots": shots,
        "input_filename": t.get("input_filename", ""),
        "output_filename": t.get("output_filename", f"pred_{task_id}.mp4"),
        "status": t.get("status", "completed")
    })


@app.route('/api/task/<task_id>/shots/<int:shot_id>', methods=['GET'])
def get_task_shot_detail(task_id, shot_id):
    """Return detailed telemetry for a single shot."""
    if task_id not in tasks:
        loaded = _load_task_telemetry(task_id)
        if loaded:
            tasks[task_id] = loaded
        else:
            return jsonify({"error": "Task not found"}), 404
    t = tasks[task_id]
    tel = t.get("telemetry") or {}
    shots = extract_or_build_shots_data(tel)
    
    matching_shot = next((s for s in shots if s.get("shot") == shot_id), None)
    if not matching_shot:
        return jsonify({"error": f"Shot {shot_id} not found"}), 404

    # Extract shot-specific trajectory points and frame data
    traj_pts = [p for p in tel.get("trajectory_points", []) if p.get("shot") == shot_id]
    frame_d = [f for f in tel.get("frame_data", []) if f.get("shot") == shot_id]

    # Best shot determination using validity filter
    valid_shots = [s for s in shots if is_valid_delivery_shot(s)]
    best_shot = None
    insights_for_shot = None
    if t.get("insights") and isinstance(t["insights"], dict) and t["insights"].get("best_shot"):
        best_shot = t["insights"].get("best_shot")
        for ish in t["insights"].get("shots", []):
            if ish.get("shot") == shot_id:
                insights_for_shot = ish
                break
    elif valid_shots:
        best_item = max(
            valid_shots,
            key=lambda s: (float(s.get("release_speed_kmh", 0) or 0) + float(s.get("max_speed_kmh", 0) or 0))
        )
        best_shot = best_item.get("shot", 1)

    matching_shot["is_best"] = (best_shot is not None and shot_id == best_shot)
    matching_shot["is_valid_delivery"] = is_valid_delivery_shot(matching_shot)
    if insights_for_shot:
        matching_shot["insights"] = insights_for_shot

    return jsonify({
        "task_id": task_id,
        "shot_id": shot_id,
        "shot": matching_shot,
        "trajectory_points": traj_pts,
        "frame_data": frame_d,
        "fps": tel.get("fps", 30.0),
        "width": tel.get("width", 1280),
        "height": tel.get("height", 720),
        "total_shots": len(shots),
        "best_shot": best_shot
    })


@app.route('/api/task/<task_id>/shots/<int:shot_id>/video', methods=['GET'])
def serve_shot_video(task_id, shot_id):
    """Serve the mp4 clip for an individual shot."""
    shot_dir = os.path.join(PROCESSED_DIR, task_id)
    shot_file = f"shot_{shot_id}.mp4"
    if os.path.exists(os.path.join(shot_dir, shot_file)):
        return send_from_directory(shot_dir, shot_file, mimetype='video/mp4', conditional=True)
    full_file = f"pred_{task_id}.mp4"
    if os.path.exists(os.path.join(PROCESSED_DIR, full_file)):
        return send_from_directory(PROCESSED_DIR, full_file, mimetype='video/mp4', conditional=True)
    return jsonify({"error": "Video not found"}), 404

# ─────────────────────────────────────────────
#  GEMINI AI INSIGHTS ENDPOINT
# Load .env if present (git-ignored)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
    except Exception:
        pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def _parse_gemini_insights(raw_text, shots_data):
    """Parse Gemini output into a clean structured dictionary with shot-wise breakdown."""
    text = (raw_text or "").strip()
    # Strip markdown code block wrappers if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    valid_shots = [s for s in shots_data if is_valid_delivery_shot(s)]
    valid_nums = {s.get("shot") for s in valid_shots}

    # Try strict json parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "shots" in data:
            for s in data.get("shots", []):
                try:
                    s["shot"] = int(s.get("shot", 1))
                except Exception:
                    pass
            try:
                cand_best = int(data.get("best_shot", 1))
                if valid_nums and cand_best not in valid_nums:
                    best_item = max(
                        valid_shots,
                        key=lambda s: (float(s.get("release_speed_kmh", 0) or 0) + float(s.get("max_speed_kmh", 0) or 0))
                    )
                    cand_best = best_item.get("shot")
                data["best_shot"] = cand_best if valid_nums else None
            except Exception:
                data["best_shot"] = next(iter(valid_nums)) if valid_nums else None
            data["has_valid_delivery"] = bool(valid_nums)
            return data
    except Exception:
        pass

    # Heuristic fallback if JSON decoding fails or Gemini returned prose
    if not valid_shots:
        return {
            "best_shot": None,
            "has_valid_delivery": False,
            "best_shot_title": "No Valid Delivery Detected",
            "verdict": "No genuine cricket delivery was detected in this clip. Tracking metrics indicate non-delivery footage or background movement (all detected motion failed delivery speed and tracking rate gates).",
            "shots": [],
            "raw_markdown": raw_text
        }

    best_shot_item = max(
        valid_shots,
        key=lambda s: (float(s.get("release_speed_kmh", 0) or 0) + float(s.get("max_speed_kmh", 0) or 0)),
        default={}
    )
    best_num = best_shot_item.get("shot", 1)

    parsed_shots = []
    for s in shots_data:
        s_num = s.get("shot", 1)
        rel = s.get("release_speed_kmh", 0)
        peak = s.get("max_speed_kmh", 0)
        b_count = s.get("bounce_count", 0)
        ang = s.get("avg_angle", 0)
        dur = s.get("duration_sec", 0)
        is_val = is_valid_delivery_shot(s)

        is_best = (s_num == best_num)
        parsed_shots.append({
            "shot": s_num,
            "title": f"Shot {s_num}: Delivery Analysis" if is_val else f"Shot {s_num}: Non-Delivery Movement",
            "delivery_type": ("Good Length Delivery" if b_count > 0 else "Full Yorker Attempt") if is_val else "Filtered Non-Delivery",
            "threat_rating": ("8/10" if is_best else "6/10") if is_val else "1/10",
            "summary": f"Delivery clocked at {rel} km/h release reaching {peak} km/h peak over {dur}s." if is_val else "Low confidence motion below genuine cricket delivery thresholds.",
            "points": [
                f"**Pace & Release**: Clocked at {rel} km/h release with {peak} km/h maximum velocity.",
                f"**Pitch & Bounce**: {b_count} pitch contact event(s) recorded at {ang}° trajectory inclination.",
                f"**Threat**: {'High challenge to batsman with sharp deck reaction.' if is_best else 'Solid control with steady line through the crease.' if is_val else 'Non-delivery movement poses no challenge.'}",
                "**Coaching Tip**: Maintain forward momentum through the delivery stride." if is_val else "No coaching guidance needed for non-delivery frames."
            ]
        })

    return {
        "best_shot": best_num,
        "has_valid_delivery": True,
        "best_shot_title": f"Shot {best_num} was the Standout Delivery",
        "verdict": f"Shot {best_num} was the most effective delivery of the spell, producing peak velocity ({best_shot_item.get('max_speed_kmh', 0)} km/h) and optimal deck impact.",
        "shots": parsed_shots,
        "raw_markdown": raw_text
    }


@app.route('/api/analyze/<task_id>', methods=['POST'])
def analyze_delivery(task_id):
    """Send delivery telemetry to Gemini and return structured shot-wise cricket coaching insights."""
    if not GEMINI_AVAILABLE:
        return jsonify({"error": "google-generativeai not installed. Run: pip install google-generativeai"}), 500

    if task_id not in tasks:
        loaded = _load_task_telemetry(task_id)
        if loaded:
            tasks[task_id] = loaded
        else:
            return jsonify({"error": "Task not found"}), 404

    t = tasks[task_id]
    if t.get("status") != "completed":
        return jsonify({"error": "Task not completed yet"}), 400

    tel = t.get("telemetry", {}) or {}
    shots_data = extract_or_build_shots_data(tel)
    tel["shots_data"] = shots_data
    valid_shots = [s for s in shots_data if is_valid_delivery_shot(s)]

    if not valid_shots:
        parsed_insights = {
            "best_shot": None,
            "has_valid_delivery": False,
            "best_shot_title": "No Valid Delivery Detected",
            "verdict": "No genuine cricket delivery was detected in this session. The detected movement did not meet cricket bowling speed, tracking rate, or kinematic delivery thresholds.",
            "shots": []
        }
        for s in shots_data:
            s["is_best"] = False
            s["is_valid_delivery"] = False

        tasks[task_id]['insights'] = parsed_insights
        tasks[task_id]['best_shot'] = None
        tasks[task_id]['has_valid_delivery'] = False
        if 'telemetry' in tasks[task_id] and tasks[task_id]['telemetry']:
            tasks[task_id]['telemetry']['shots_data'] = shots_data
            tasks[task_id]['telemetry']['shots'] = shots_data
            tasks[task_id]['telemetry']['best_shot'] = None
            tasks[task_id]['telemetry']['has_valid_delivery'] = False
        _add_to_history(task_id, tasks[task_id])

        return jsonify({
            "task_id": task_id,
            "has_valid_delivery": False,
            "insights": parsed_insights,
            "shots_data": shots_data,
            "data_summary": {
                "release_kmh": 0.0,
                "avg_kmh": 0.0,
                "peak_kmh": 0.0,
                "bounce_count": 0,
                "total_shots": len(shots_data),
                "valid_shots": 0,
                "best_shot": None
            }
        })

    speed = tel.get("speed_summary", {}) or {}
    bounces = tel.get("bounce_events", []) or []
    total_frames = tel.get("total_frames", 0)
    fps = tel.get("fps", 30)
    total_shots = len(shots_data)

    release_kmh = speed.get("release_speed_kmh", 0)
    avg_kmh = speed.get("avg_speed_kmh", 0)
    peak_kmh = speed.get("max_speed_kmh", 0)

    # Format shot summaries for the prompt (prioritizing genuine deliveries)
    shot_prompts = []
    for s in shots_data:
        s_num = s["shot"]
        is_val = is_valid_delivery_shot(s)
        b_list = s.get("bounces", [])
        b_desc = ", ".join([f"Frame {b.get('frame')} @ {b.get('angle')}° ({b.get('speed_kmh')} km/h)" for b in b_list]) if b_list else "None (full flight)"
        shot_prompts.append(
            f"SHOT {s_num}{' (Genuine Delivery)' if is_val else ' (Low Confidence / Non-Delivery Motion)'}:\n"
            f"  - Valid Delivery: {is_val}\n"
            f"  - Timing: {s.get('start_sec')}s - {s.get('end_sec')}s ({s.get('duration_sec')}s duration)\n"
            f"  - Release Speed: {s.get('release_speed_kmh')} km/h ({s.get('release_speed_mph')} mph)\n"
            f"  - Avg Speed: {s.get('avg_speed_kmh')} km/h | Peak Speed: {s.get('max_speed_kmh')} km/h\n"
            f"  - Pitch Bounces ({len(b_list)}): {b_desc}\n"
            f"  - Trajectory Angle: {s.get('avg_angle')}°"
        )
    shots_summary_text = "\n\n".join(shot_prompts)

    try:
        prompt = f"""You are HawkVision AI, an elite cricket bowling performance analyst and biomechanics coach.
Analyze this video session containing {total_shots} cricket delivery shot(s).
Important: Only genuine deliveries marked Valid Delivery: True should be chosen as best shot.

=== TELEMETRY BY SHOT ===
{shots_summary_text}

=== INSTRUCTIONS ===
1. Compare all shots and determine the **BEST SHOT** (the most effective delivery in terms of pace, length, bounce, and batsman difficulty).
2. Keep the report SHORT, CRISP, and TO THE POINT. Use concise bullet points. Avoid unnecessary fluff or conversational padding.
3. For EACH shot, provide:
   - delivery_type: e.g. "Good Length Skidder", "Full Yorker", "Short-Pitched Bouncer"
   - threat_rating: e.g. "8/10"
   - summary: 1 crisp sentence explaining this shot.
   - points: exactly 3-4 punchy bullet points covering:
     * **Pace & Release**: speed analysis
     * **Pitch & Bounce**: length and angle off the pitch
     * **Threat**: challenge posed to batsman
     * **Coaching Tip**: 1 actionable adjustment
4. Provide a 2-sentence comparative **verdict** explaining which shot was best and why.
5. Return ONLY a valid JSON object matching this schema (do NOT write markdown outside the JSON):
{{
  "best_shot": 1,
  "best_shot_title": "Shot 1 was the Best Delivery",
  "verdict": "Shot 1 was the most penetrative delivery due to high kinetic retention (138.2 km/h bounce impact) and sharp 28.5 degree skid off the pitch, keeping the batter under pressure compared to Shot 2.",
  "shots": [
    {{
      "shot": 1,
      "title": "Shot 1: Rapid Good-Length Skidder",
      "delivery_type": "Good Length Skidder",
      "threat_rating": "8/10",
      "summary": "Fastest ball with sharp skidding bounce.",
      "points": [
        "**Pace**: Strong release reaching 138.2 km/h kinetic peak off pitch.",
        "**Pitch & Bounce**: Low 28.5° skid keeping batsman hurried on back foot.",
        "**Threat**: High threat (8/10) due to late trajectory drop.",
        "**Coaching Tip**: Maintain this upright wrist seam position."
      ]
    }}
  ]
}}

CRITICAL FORMATTING RULE: NEVER use LaTeX math or dollar signs (do NOT use dollar signs or backslash notation). Always write natural numbers, units, and symbols directly (e.g. 28.5 degrees, 138.2 km/h)."""

        genai.configure(api_key=GEMINI_API_KEY)
        candidate_models = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash"]
        raw_insights = None
        last_err = None

        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    raw_insights = response.text
                    break
            except Exception as m_err:
                last_err = m_err
                continue

        if not raw_insights:
            raise RuntimeError(f"All candidate models failed. Last error: {last_err}")

        parsed_insights = _parse_gemini_insights(raw_insights, shots_data)

        # Update each shot's insights and is_best flag
        best_shot_idx = parsed_insights.get("best_shot")
        insights_by_shot = {s.get("shot"): s for s in parsed_insights.get("shots", [])}
        for s in shots_data:
            s["is_best"] = (best_shot_idx is not None and s.get("shot") == best_shot_idx)
            s["is_valid_delivery"] = is_valid_delivery_shot(s)
            if s.get("shot") in insights_by_shot:
                s["insights"] = insights_by_shot[s.get("shot")]

        # Save insights into task + persist history
        tasks[task_id]['insights'] = parsed_insights
        tasks[task_id]['best_shot'] = best_shot_idx
        tasks[task_id]['has_valid_delivery'] = bool(valid_shots)
        if 'telemetry' in tasks[task_id] and tasks[task_id]['telemetry']:
            tasks[task_id]['telemetry']['shots_data'] = shots_data
            tasks[task_id]['telemetry']['shots'] = shots_data
            tasks[task_id]['telemetry']['best_shot'] = best_shot_idx
            tasks[task_id]['telemetry']['has_valid_delivery'] = bool(valid_shots)
        _add_to_history(task_id, tasks[task_id])

        return jsonify({
            "task_id": task_id,
            "has_valid_delivery": bool(valid_shots),
            "insights": parsed_insights,
            "shots_data": shots_data,
            "data_summary": {
                "release_kmh": release_kmh,
                "avg_kmh": avg_kmh,
                "peak_kmh": peak_kmh,
                "bounce_count": len(bounces),
                "total_shots": total_shots,
                "valid_shots": len(valid_shots),
                "best_shot": best_shot_idx
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 500


# ─────────────────────────────────────────────
#  HISTORY API ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
def get_history():
    """Return all processed delivery history records, newest first."""
    records = _load_history_index()
    # Filter out entries whose output video file no longer exists
    valid = []
    for r in records:
        vpath = os.path.join(PROCESSED_DIR, r.get('output_filename', ''))
        if os.path.exists(vpath):
            valid.append(r)
    return jsonify({"history": valid, "count": len(valid)})


@app.route('/api/history/<task_id>', methods=['GET'])
def get_history_item(task_id):
    """Return full task payload + telemetry for a historical delivery."""
    # Try in-memory cache first (if task is still in current session)
    if task_id in tasks:
        t = tasks[task_id]
        if not t.get("task_id"):
            t["task_id"] = task_id
        if t.get("telemetry"):
            tel = t["telemetry"]
            if "shots_data" not in tel or not tel["shots_data"]:
                tel["shots_data"] = extract_or_build_shots_data(tel)
        return jsonify(t)
    # Fall back to persisted telemetry JSON
    data = _load_task_telemetry(task_id)
    if data:
        if not data.get("task_id"):
            data["task_id"] = task_id
        if data.get("telemetry"):
            tel = data["telemetry"]
            if "shots_data" not in tel or not tel["shots_data"]:
                tel["shots_data"] = extract_or_build_shots_data(tel)
        tasks[task_id] = data
        return jsonify(data)
    # Minimal fallback from index
    records = _load_history_index()
    record = next((r for r in records if r.get('task_id') == task_id), None)
    if record:
        return jsonify({
            'task_id': task_id,
            'status': 'completed',
            'output_filename': record.get('output_filename', f'pred_{task_id}.mp4'),
            'input_filename': record.get('input_filename', ''),
            'timestamp': record.get('timestamp', ''),
            'telemetry': None
        })
    return jsonify({"error": "History record not found"}), 404


@app.route('/api/history/<task_id>', methods=['DELETE'])
def delete_history_item(task_id):
    """Remove a history record (and optionally its video/json files)."""
    records = _load_history_index()
    record = next((r for r in records if r.get('task_id') == task_id), None)
    if not record:
        return jsonify({"error": "Not found"}), 404

    delete_files = request.args.get('delete_files', 'false').lower() == 'true'
    if delete_files:
        for fname in [record.get('output_filename', ''), f'pred_{task_id}.json']:
            fpath = os.path.join(PROCESSED_DIR, fname)
            if fname and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f'[History] Could not delete {fpath}: {e}')

    records = [r for r in records if r.get('task_id') != task_id]
    _save_history_index(records)
    if task_id in tasks:
        del tasks[task_id]

    return jsonify({"success": True, "task_id": task_id})


@app.route('/history')
def history_page():
    """Dedicated session history page exactly matching cricket-copy design."""
    records = _load_history_index()
    valid_sessions = []
    for r in records:
        vpath = os.path.join(PROCESSED_DIR, r.get('output_filename', ''))
        if os.path.exists(vpath):
            ts = r.get('timestamp', '')
            formatted_date = ts
            try:
                dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%b %d, %Y • %I:%M %p')
            except Exception:
                pass

            total = r.get('total_frames') or 0
            detected = r.get('detected_frames') or 0
            pct = max(0, min(100, int((detected / max(total, 1)) * 100))) if total > 0 else 0

            # Classification category based on speed & trajectory
            rel_spd = r.get('release_speed_kmh', 0)
            avg_spd = r.get('avg_speed_kmh', 0)
            eff_spd = rel_spd or avg_spd
            if eff_spd >= 135:
                mode_label = "Fast Delivery"
            elif eff_spd >= 115:
                mode_label = "Medium Pace"
            elif eff_spd > 0:
                mode_label = "Spin / Slow"
            else:
                mode_label = "Delivery Action"

            valid_sessions.append({
                'task_id': r.get('task_id'),
                'job_id': r.get('task_id'),
                'created_at': formatted_date,
                'timestamp': ts,
                'input_filename': r.get('input_filename') or f"Delivery #{r.get('task_id')}",
                'output_filename': r.get('output_filename'),
                'video_filename': r.get('output_filename'),
                'duration_sec': r.get('duration_sec', 0.0),
                'fps': r.get('fps', 30.0),
                'total_frames': total,
                'detected_frames': detected,
                'score_pct': pct,
                'mode': mode_label,
                'release_speed_kmh': round(rel_spd, 1) if rel_spd else 0,
                'avg_speed_kmh': round(avg_spd, 1) if avg_spd else 0,
                'max_speed_kmh': round(r.get('max_speed_kmh', 0), 1),
                'bounce_count': r.get('bounce_count', 0),
                'total_shots': r.get('total_shots', 1),
                'has_insights': bool(r.get('has_insights'))
            })

    return render_template('history.html', sessions=valid_sessions)


@app.route('/delete/<task_id>', methods=['POST', 'DELETE'])
def delete_session_alias(task_id):
    """Compatibility delete route matching cricket-copy."""
    res = delete_history_item(task_id)
    if isinstance(res, tuple):
        resp, code = res
        return jsonify({'status': 'error', 'error': 'Not found'}), code
    data = res.get_json() if hasattr(res, 'get_json') else {}
    if data.get('success'):
        return jsonify({'status': 'ok', 'success': True, 'task_id': task_id})
    return jsonify({'status': 'error', 'error': 'Could not delete'}), 500


@app.route('/results/video/<path:filename>')
def serve_results_video(filename):
    """Video serve route matching cricket-copy pattern."""
    return send_from_directory(PROCESSED_DIR, filename, mimetype='video/mp4', conditional=True)


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear all history records from the index."""
    _save_history_index([])
    return jsonify({"success": True, "message": "History cleared"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8088))
    print(f"🚀 Cricket Ball Trajectory Web App running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
