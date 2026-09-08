"""
HawkVision Multi-Delivery Segmentation Validation Script.
Usage:
    python test_segmentation.py [optional_video_path]
"""

import sys
import os
import json
from trajectory_tracker import segment_deliveries

def main():
    default_video = os.path.join('videos', 'upload_e7773d3f_Screen Recording 2026-09-02 153903.mp4')
    if not os.path.exists(default_video):
        default_video = os.path.join('videos', 'test1.mp4')

    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"🏏 Testing HawkVision Two-Stage Segmentation Pipeline")
    print(f"📹 Target Video: {video_path}")
    print(f"=======================================================\n")

    task_id = "test_validate"
    result = segment_deliveries(video_path, task_id=task_id, debug=True)

    shots = result.get("shots", [])
    shot_count = result.get("shot_count", 0)

    print(f"\n🎯 SUMMARY RESULTS:")
    print(f"Total Video Frames: {result.get('total_video_frames')} @ {result.get('fps')} FPS")
    print(f"Total Detected Deliveries: {shot_count}\n")

    for s in shots:
        print(f"  • Shot {s['shot_index']}:")
        print(f"      Frame Range : {s['start_frame']} -> {s['end_frame']} ({s['duration_sec']}s)")
        print(f"      Time Window : {s['start_time_sec']}s -> {s['end_time_sec']}s")
        print(f"      Confidence  : {s['segmentation_confidence'] * 100:.1f}%")
        print(f"      Parabola R² : {s['parabola_r2']}")
        print(f"      Points Track: {s['detected_points']} frames")
        print(f"      Clip Target : {s['clip_path']}")
        print()

    debug_file = os.path.join('videos', 'processed', f"{task_id}_segmentation_debug.json")
    if os.path.exists(debug_file):
        print(f"📝 Debug segmentation log dumped to: {debug_file}")

    print("\n✅ Segmentation sanity check complete!\n")

if __name__ == '__main__':
    main()
