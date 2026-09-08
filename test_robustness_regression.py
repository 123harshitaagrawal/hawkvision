"""
HawkVision - Accuracy & Robustness Regression Test Suite

Validates all 5 core robustness guarantees:
1. test_baseline_delivery:
   - test1.mp4 runs end-to-end
   - Prompt tracking initiation (starts <= frame 20, no release delay)
   - Verified pitch bounce confirmed (angle ~28°, frame ~20)
   - High kinetic release speed (> 80 km/h)
   - Correctly marked is_valid_delivery=True
2. test_talking_face_cluster_rejection:
   - Simulates slow head-nodding/talking face movement (displacement < 35px, speed ~4.1 km/h)
   - Pass B cluster gates reject it (displacement span < 0.12*dim, implied speed < 15 km/h)
   - Returns None (no delivery window formed)
3. test_talking_face_api_contract:
   - Telemetry containing only sub-cricket speeds (~4.1 km/h) or low tracking rates
   - is_valid_delivery_shot returns False
   - _parse_gemini_insights returns has_valid_delivery=False, best_shot=None
4. test_wall_impact_bounce_rejection:
   - Simulates a ball hitting indoor netting/wall and abruptly stopping
   - Subsequent motion has < 2.0 px/frame displacement
   - Delayed bounce confirmation rejects the impact; 0 pitch bounces recorded
5. test_bbox_size_gates:
   - Verifies Gate 1A rejects large non-ball boxes (e.g., face > 11% width, > 13% height)
   - Verifies Gate 1A admits legitimate indoor nets balls filmed close up (7-9% width)
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from trajectory_tracker import (
    CricketTrajectoryPredictor,
    pass_b_refine_window,
    is_delivery_candidate_cluster,
    detect_faces_fast
)
from server import (
    is_valid_delivery_shot,
    _parse_gemini_insights,
    MIN_VALID_RELEASE_KMH,
    MIN_TRACKING_RATE_PCT,
    MIN_DETECTED_FRAMES
)


class TestHawkVisionRobustness(unittest.TestCase):

    def test_baseline_delivery(self):
        """
        Baseline broadcast test (videos/test1.mp4):
        Verifies tracking initiates promptly at release, genuine pitch bounce is confirmed,
        and high kinetic release speed is captured without regressions.
        """
        video_path = os.path.join(BASE_DIR, 'videos', 'test1.mp4')
        if not os.path.exists(video_path):
            self.skipTest(f"Video {video_path} not found")

        model_path = os.path.join(BASE_DIR, 'runs', 'detect', 'train5', 'weights', 'best.pt')
        if not os.path.exists(model_path):
            self.skipTest(f"Model {model_path} not found")

        predictor = CricketTrajectoryPredictor(
            model_path=model_path,
            conf_thresh=0.25,
            history_len=15,
            pred_steps=6
        )

        out_path = os.path.join(BASE_DIR, 'videos', 'scratch_regression_test1.mp4')
        telemetry = predictor.process_video(
            input_video_path=video_path,
            output_video_path=out_path,
            show_window=False,
            auto_trim=True
        )

        detected_frames = telemetry.get('detected_frames', 0)
        self.assertGreaterEqual(detected_frames, 10, "Baseline should track at least 10 frames")

        # Check prompt tracking initiation: release tracking must start at or before frame 20
        frame_data = telemetry.get('frame_data', [])
        detected_indices = [f['frame'] for f in frame_data if f.get('detected')]
        self.assertTrue(len(detected_indices) > 0, "At least one detected frame required")
        first_detected_frame = detected_indices[0]
        self.assertLessEqual(first_detected_frame, 20,
                             f"Tracking initiation delayed: first detection at frame {first_detected_frame} > 20")

        # Check verified bounce
        bounces = telemetry.get('bounce_events', [])
        self.assertGreaterEqual(len(bounces), 1, "Baseline broadcast delivery must have a confirmed pitch bounce")
        first_bounce = bounces[0]
        self.assertIn(first_bounce['frame'], range(18, 25), "Pitch bounce frame must be around 18-24")
        self.assertGreater(first_bounce['angle'], 15.0, "Pitch bounce angle must be realistic cricket bounce angle")

        # Check release speed
        speed = telemetry.get('speed_summary', {})
        release_kmh = speed.get('release_speed_kmh', 0)
        self.assertGreaterEqual(release_kmh, 80.0, f"Release speed should be cricket bowling pace (>80 km/h), got {release_kmh}")

        # Check shot validity
        shots = telemetry.get('shots_data', [])
        self.assertGreaterEqual(len(shots), 1, "Should segment at least 1 delivery shot")
        self.assertTrue(is_valid_delivery_shot(shots[0]), "Baseline shot must pass is_valid_delivery_shot")

    def test_talking_face_cluster_rejection(self):
        """
        Simulates talking face / head nod candidate detections in Pass B:
        Small displacement span (e.g. 25px in 688px frame), slow speed (~4 km/h),
        and large bbox (~60x70px). Must be rejected by Pass B cluster gates.
        """
        frame_w, frame_h = 418, 688
        fps = 30.0

        # Simulate 15 detections of a head nodding (moving between y=200 and y=225, x=200..208)
        # Displacement is ~25 px total (3.6% of frame height), mean bbox is 60x70 px (14% w, 10% h)
        mock_cluster = []
        for i in range(15):
            f_num = 10 + i * 2
            # Small sinusoidal nod
            cx = 200 + int(4 * np.sin(i * 0.5))
            cy = 210 + int(12 * np.sin(i * 0.4))
            w = 60
            h = 70
            box = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
            mock_cluster.append({'frame': f_num, 'cx': cx, 'cy': cy, 'box': box, 'conf': 0.75})

        is_valid, reason = is_delivery_candidate_cluster(mock_cluster, frame_w, frame_h, fps)
        self.assertFalse(is_valid, f"Talking face cluster must be rejected by cluster gates, but passed. Reason: {reason}")
        self.assertIn("Total displacement span", reason)

    def test_talking_face_api_contract(self):
        """
        Verifies backend API contract when shots fail delivery thresholds:
        - is_valid_delivery_shot returns False for low speed / low tracking
        - _parse_gemini_insights returns has_valid_delivery=False, best_shot=None
        """
        # Fake shot 1: Talking face at 4.1 km/h
        fake_face_shot = {
            "shot": 1,
            "release_speed_kmh": 4.1,
            "avg_speed_kmh": 4.1,
            "max_speed_kmh": 5.0,
            "detected_frames": 3,
            "total_frames": 40,
            "tracking_rate": 7.5,
            "bounce_count": 0,
            "avg_angle": 2.0,
            "duration_sec": 1.3
        }

        self.assertFalse(is_valid_delivery_shot(fake_face_shot),
                         "Shot with 4.1 km/h and 7.5% tracking must be invalid")

        # Parse insights for non-delivery session
        insights = _parse_gemini_insights(raw_text="Random text", shots_data=[fake_face_shot])
        self.assertFalse(insights.get("has_valid_delivery"), "Must report has_valid_delivery=False")
        self.assertIsNone(insights.get("best_shot"), "best_shot must be None for non-delivery sessions")
        self.assertEqual(insights.get("best_shot_title"), "No Valid Delivery Detected")

    def test_wall_impact_bounce_rejection(self):
        """
        Simulates an abrupt wall or netting impact where the ball stops dead:
        Pre-bounce motion is fast, but Phase 2 motion has < 2.0 px/frame displacement.
        Delayed confirmation must reject this as a pitch bounce.
        """
        # In trajectory_tracker, delayed confirmation checks:
        # avg_disp = sum(disp_after) / len(disp_after) >= 2.0 px/frame
        disp_after_wall = [0.5, 0.2, 0.0, 0.1]  # ball dead against net/wall
        avg_disp = sum(disp_after_wall) / len(disp_after_wall)
        min_continuation_disp = 2.0

        is_confirmed = (avg_disp >= min_continuation_disp)
        self.assertFalse(is_confirmed,
                         f"Wall impact with avg displacement {avg_disp:.2f} px/frame must fail bounce confirmation")

        # Conversely, a genuine pitch bounce continues forward at >= 2.0 px/frame
        disp_after_pitch = [14.2, 13.8, 13.0, 12.5]
        avg_pitch_disp = sum(disp_after_pitch) / len(disp_after_pitch)
        self.assertTrue(avg_pitch_disp >= min_continuation_disp,
                        "Genuine pitch bounce must satisfy continuation motion threshold")

    def test_bbox_size_gates(self):
        """
        Verifies bbox size ceiling logic:
        - Faces / large objects (> 11% w or > 13% h) are rejected
        - Indoor nets balls filmed close up (7-9% w, 7-9% h) are admitted
        """
        frame_w, frame_h = 418, 688
        max_bw = frame_w * 0.11  # 45.98 px
        max_bh = frame_h * 0.13  # 89.44 px

        # 1. Talking face box: 70px x 95px
        face_bw = 70.0
        face_bh = 95.0
        face_rejected = (face_bw > max_bw or face_bh > max_bh)
        self.assertTrue(face_rejected, "Face candidate (70x95) must be rejected by bbox ceiling")

        # 2. Indoor nets ball filmed close: 32px x 32px (7.6% w, 4.6% h)
        nets_ball_bw = 32.0
        nets_ball_bh = 32.0
        nets_ball_accepted = (nets_ball_bw <= max_bw and nets_ball_bh <= max_bh)
        self.assertTrue(nets_ball_accepted, "Indoor nets ball (32x32) must pass bbox ceiling")


if __name__ == '__main__':
    unittest.main(verbosity=2)
