import os
import argparse
import sys
from trajectory_tracker import CricketTrajectoryPredictor

def main():
    parser = argparse.ArgumentParser(description="Cricket Ball Trajectory Prediction using YOLOv8")
    parser.add_argument("--video", "-v", type=str, default="videos/test1.mp4", 
                        help="Path to input cricket video (default: videos/test1.mp4)")
    parser.add_argument("--output", "-o", type=str, default="videos/output_predicted.mp4",
                        help="Path to save annotated trajectory video (default: videos/output_predicted.mp4)")
    parser.add_argument("--model", "-m", type=str, default="runs/detect/train5/weights/best.pt",
                        help="Path to trained YOLOv8 model weights")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="Confidence threshold for ball detection (default: 0.25)")
    parser.add_argument("--history", type=int, default=15,
                        help="Trajectory history length in frames (default: 15)")
    parser.add_argument("--future", type=int, default=6,
                        help="Future trajectory prediction steps (default: 6)")
    parser.add_argument("--bezier", action="store_true", default=True,
                        help="Enable Bezier smoothing for past trajectory")
    parser.add_argument("--persistent", action="store_true", default=True,
                        help="Draw persistent full delivery trajectory ribbon (default: True)")
    parser.add_argument("--trim", dest="auto_trim", action="store_true", default=True,
                        help="Auto-trim video to the active delivery window (default: True)")
    parser.add_argument("--no-trim", dest="auto_trim", action="store_false",
                        help="Disable auto-trim and process entire video")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Target compute device: '0' for CUDA GPU, 'cpu' for CPU (default: auto-detect)")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Disable live OpenCV display window")
    
    args = parser.parse_args()

    # Detect Google Colab / Headless Linux environment
    is_colab = 'google.colab' in sys.modules
    is_headless = is_colab or (sys.platform.startswith('linux') and not os.environ.get('DISPLAY'))
    if is_headless:
        args.no_show = True

    # Detect GPU / CUDA status
    import torch
    cuda_avail = torch.cuda.is_available()
    if args.device is None:
        target_device = 0 if cuda_avail else 'cpu'
    else:
        target_device = args.device

    device_label = f"CUDA GPU ({torch.cuda.get_device_name(0)})" if (cuda_avail and str(target_device) != 'cpu') else "CPU"

    # Resolve relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    video_path = args.video
    if not os.path.isabs(video_path):
        if not os.path.exists(video_path) and os.path.exists(os.path.join(base_dir, video_path)):
            video_path = os.path.join(base_dir, video_path)

    model_path = args.model
    if not os.path.isabs(model_path):
        if not os.path.exists(model_path) and os.path.exists(os.path.join(base_dir, model_path)):
            model_path = os.path.join(base_dir, model_path)
        elif not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "runs", "detect", "train5", "weights", "best.pt")

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    print(f"============================================================")
    print(f"🏏 CRICKET BALL TRAJECTORY PREDICTION SYSTEM (HAWKVISION)")
    print(f"============================================================")
    print(f"Compute Device: {device_label}")
    print(f"Input Video   : {video_path}")
    print(f"Model Weights : {model_path}")
    print(f"Confidence    : {args.conf}")
    print(f"Auto-Trim     : {'Enabled' if args.auto_trim else 'Disabled'}")
    print(f"Output Video  : {output_path}")
    print(f"Live Window   : {'Disabled (Headless/Colab)' if args.no_show else 'Enabled (Press Q to quit)'}")
    print(f"============================================================\n")

    if not os.path.exists(video_path):
        print(f"Error: Input video not found at '{video_path}'")
        sys.exit(1)

    predictor = CricketTrajectoryPredictor(
        model_path=model_path,
        conf_thresh=args.conf,
        history_len=args.history,
        pred_steps=args.future,
        device=target_device
    )

    def on_progress(frame, total, angle, detected):
        status = "BALL TRACKED" if detected else "SEARCHING"
        sys.stdout.write(f"\rProcessing frame [{frame}/{total}] ({int(frame/total*100)}%) | Status: {status} | Angle: {angle:.1f}°")
        sys.stdout.flush()

    telemetry = predictor.process_video(
        input_video_path=video_path,
        output_video_path=output_path,
        show_window=(not args.no_show),
        use_bezier=args.bezier,
        persistent_trail=args.persistent,
        auto_trim=args.auto_trim,
        progress_callback=on_progress
    )

    print(f"\n\n============================================================")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"============================================================")
    print(f"Total Frames Processed : {telemetry['total_frames']}")
    if telemetry.get("total_shots", 1) > 1:
        print(f"🏏 Multi-Shot Auto-Trim: Combined {telemetry['total_shots']} delivery shots into one video!")
    print(f"Ball Detected In       : {telemetry['detected_frames']} frames")
    if telemetry.get("speed_summary"):
        spd = telemetry["speed_summary"]
        print(f"🏏 BALL SPEED KINEMATICS:")
        print(f"  - Release Speed      : {spd['release_speed_kmh']} km/h ({spd['release_speed_mph']} mph)")
        print(f"  - Average Flight Spd : {spd['avg_speed_kmh']} km/h ({spd['avg_speed_mph']} mph)")
        print(f"  - Peak Recorded Spd  : {spd['max_speed_kmh']} km/h ({spd['max_speed_mph']} mph)")
    print(f"Bounce Events Detected : {len(telemetry['bounce_events'])}")
    for i, b in enumerate(telemetry['bounce_events'], 1):
        b_spd = f" | Speed: {b.get('speed_kmh', '--')} km/h" if 'speed_kmh' in b else ""
        print(f"  - Bounce #{i}: Frame {b['frame']} @ {b['timestamp']}s (Angle: {b['angle']}°{b_spd})")
    print(f"Output Video Saved To  : {output_path}")
    print(f"============================================================\n")

if __name__ == "__main__":
    main()
