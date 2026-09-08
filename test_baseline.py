import json
from trajectory_tracker import CricketTrajectoryPredictor

predictor = CricketTrajectoryPredictor()
telemetry = predictor.process_video('videos/test1.mp4', 'videos/scratch_test1_out.mp4', task_id='scratch_test1')

print('Detected frames:', telemetry.get('detected_frames'))
print('Bounce events:', telemetry.get('bounce_events'))
traj = telemetry.get('trajectory_points', [])
print('Total trajectory points:', len(traj))

phase1 = [p for p in traj if p.get('phase') == 1]
phase2 = [p for p in traj if p.get('phase') == 2]
print(f"Phase 1 points ({len(phase1)}):", [(p['frame'], (p['x'], p['y'])) for p in phase1[:4]], '...', [(p['frame'], (p['x'], p['y'])) for p in phase1[-3:]])
print(f"Phase 2 points ({len(phase2)}):", [(p['frame'], (p['x'], p['y'])) for p in phase2[:4]], '...', [(p['frame'], (p['x'], p['y'])) for p in phase2[-3:]])

for s in telemetry.get('shots_data', []):
    print(f"Shot {s.get('shot')}: bounds={s.get('start_frame')}..{s.get('end_frame')}, bounce={s.get('bounce_point')}, stump_hit={s.get('stump_hit_point')}, is_stump={s.get('is_stump_hit')}")
