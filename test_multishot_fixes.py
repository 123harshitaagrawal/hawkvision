import json
import math
from trajectory_tracker import find_delivery_windows, CricketTrajectoryPredictor

video_path = 'videos/upload_e7773d3f_Screen Recording 2026-09-02 153903.mp4'
predictor = CricketTrajectoryPredictor()

print("1. Running find_delivery_windows...")
windows, trimmed = find_delivery_windows(video_path, predictor.model)
print("Delivery windows:", windows, "trimmed:", trimmed)

print("2. Running process_video...")
telemetry = predictor.process_video(video_path, 'videos/scratch_e777_out.mp4', task_id='scratch_e777')
print("Total shots:", telemetry.get('total_shots'))

has_v_hook = False
for s in telemetry.get('shots_data', []):
    s_idx = s.get('shot')
    start_f = s.get('start_frame')
    end_f = s.get('end_frame')
    p1 = s.get('phase_1_trajectory', [])
    p2 = s.get('phase_2_trajectory', [])
    bounce = s.get('bounce_point')
    stump = s.get('stump_hit_point')
    is_stump = s.get('is_stump_hit')
    print(f"\n--- Shot {s_idx} ---")
    print(f"Bounds: frame {start_f} to {end_f} (duration: {s.get('duration_sec')}s)")
    print(f"Phase 1 count: {len(p1)}, Phase 2 count: {len(p2)}")
    print(f"Bounce point: {bounce}, Stump hit: {stump}, Is stump hit: {is_stump}")

    # Check for backward jump / V-hook in all trajectory points of this shot
    shot_pts = [p for p in telemetry.get('trajectory_points', []) if p.get('shot') == s_idx]
    max_backward_y = 0
    for i in range(1, len(shot_pts)):
        dy = shot_pts[i]['y'] - shot_pts[i-1]['y']
        frame_gap = shot_pts[i]['frame'] - shot_pts[i-1]['frame']
        if dy < -70 and frame_gap > 5:
            print(f"WARNING: V-HOOK / BACKWARD JUMP DETECTED in Shot {s_idx}! Frame {shot_pts[i-1]['frame']} -> {shot_pts[i]['frame']}, dy={dy}")
            has_v_hook = True

if not has_v_hook:
    print("\nSUCCESS: ZERO V-hook / cross-shot trajectory bleed detected across all shots!")
