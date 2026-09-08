import math
from collections import deque
import numpy as np

class CricketSpeedTracker:
    """
    Cricket Ball Speed Tracking & Kinematics Engine.
    
    Estimates instantaneous ball velocity, delivery release speed, pitch bounce speed,
    and average delivery speed in both km/h and mph using real-world pitch geometry (20.12m/17.68m)
    and camera perspective scaling without modifying the underlying YOLO detection models.
    """

    # Physical constants of cricket pitch (meters)
    PITCH_STUMP_TO_STUMP_M = 20.12       # 22 yards
    BOWLING_TO_POPPING_CREASE_M = 17.68  # Distance between bowling crease and batting popping crease
    EFFECTIVE_FLIGHT_DISTANCE_M = 18.2   # Typical ball flight distance from bowler release to batsman

    def __init__(self, fps=30.0, frame_width=640, frame_height=360, pitch_length_m=None):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width = int(frame_width)
        self.height = int(frame_height)
        self.pitch_length_m = float(pitch_length_m) if pitch_length_m else self.EFFECTIVE_FLIGHT_DISTANCE_M

        # State tracking
        self.all_deliveries = []   # summaries of previous deliveries in multi-delivery video
        self.tracked_points = []   # list of (frame_idx, x, y, is_estimated)
        self.instant_speeds_kmh = []
        self.smoothed_speeds_kmh = []
        self.last_speed_kmh = 0.0
        self.current_speed_mph = 0.0

        # Kinematics milestones
        self.release_speed_kmh = None
        self.bounce_speeds = []     # speeds at detected bounce events
        self.max_speed_kmh = 0.0
        self.avg_speed_kmh = 0.0

        # Perspective parameters (calibrated dynamically across trajectory span)
        self._scale_m_per_px_base = None
        self._delivery_axis = 'y'   # 'y' for standard behind-the-bowler broadcast, 'x' for side-on

    def reset_delivery(self):
        """
        Reset tracker state for a new delivery shot in a multi-delivery video,
        saving the completed delivery summary to all_deliveries.
        """
        if len(self.tracked_points) >= 3:
            try:
                curr_summary = self.get_summary(include_history=False)
                self.all_deliveries.append(curr_summary)
            except Exception:
                pass

        self.tracked_points = []
        self.instant_speeds_kmh = []
        self.smoothed_speeds_kmh = []
        self.last_speed_kmh = 0.0
        self.current_speed_mph = 0.0
        self.release_speed_kmh = None
        self.bounce_speeds = []
        self.max_speed_kmh = 0.0
        self.avg_speed_kmh = 0.0
        self._scale_m_per_px_base = None

    def _estimate_dynamic_scale(self):
        """
        Dynamically estimate perspective meters-per-pixel scaling based on tracked trajectory span.
        """
        if len(self.tracked_points) < 4:
            # Default heuristic based on standard 1080p/720p/360p broadcast framing
            # Pitch usually covers ~50-70% of frame height in broadcast view
            pitch_px_estimate = self.height * 0.60
            return self.pitch_length_m / max(pitch_px_estimate, 50.0)

        xs = [pt[1] for pt in self.tracked_points]
        ys = [pt[2] for pt in self.tracked_points]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        self._delivery_axis = 'y' if span_y >= span_x else 'x'
        dominant_span = max(span_y, span_x)
        
        # Flight span corresponds to release-to-impact flight
        flight_px = max(dominant_span, self.height * 0.40)
        return self.pitch_length_m / max(flight_px, 30.0)

    def update(self, frame_idx, centroid, is_estimated=False):
        """
        Update speed tracker with a new ball centroid (x, y) at given frame_idx.
        Returns (speed_kmh, speed_mph).
        """
        if centroid is None:
            return float(self.last_speed_kmh), float(self.current_speed_mph)

        cx, cy = float(centroid[0]), float(centroid[1])
        self.tracked_points.append((frame_idx, cx, cy, is_estimated))

        if len(self.tracked_points) < 2:
            return 0.0, 0.0

        prev_frame, px, py, _ = self.tracked_points[-2]
        delta_frames = max(frame_idx - prev_frame, 1)
        delta_t = delta_frames / self.fps

        # Pixel displacement
        dx = cx - px
        dy = cy - py
        dist_px = math.hypot(dx, dy)

        # Update dynamic scale if needed
        scale = self._estimate_dynamic_scale()

        # Perspective adjustment: objects at the bowler's end (higher in frame for standard broadcast)
        # appear smaller, so 1 pixel represents slightly more physical distance than near the batsman.
        norm_y = cy / max(self.height, 1)
        perspective_factor = 1.0 + (0.35 * (1.0 - norm_y))  # ~1.0 to 1.35 scaling gradient
        dist_m = (dist_px * scale) * perspective_factor

        # Raw instantaneous velocity (m/s -> km/h)
        speed_raw_kmh = (dist_m / delta_t) * 3.6 if delta_t > 0 else 0.0

        # Physical ceiling clamp for cricket ball flight (max realistic speed is ~165 km/h / ~102.5 mph)
        speed_clamped_kmh = max(0.0, min(165.0, speed_raw_kmh))

        # Exponential Moving Average (EMA) smoothing to eliminate single-frame pixel jitter
        alpha = 0.60
        if self.smoothed_speeds_kmh:
            smooth_kmh = (alpha * speed_clamped_kmh) + ((1.0 - alpha) * self.smoothed_speeds_kmh[-1])
        else:
            smooth_kmh = speed_clamped_kmh

        # Update running stats
        self.instant_speeds_kmh.append(speed_clamped_kmh)
        self.smoothed_speeds_kmh.append(smooth_kmh)
        self.last_speed_kmh = float(round(smooth_kmh, 1))
        self.current_speed_mph = float(round(self.last_speed_kmh * 0.621371, 1))

        # Capture Release Speed in the initial flight window (frames 2 to 6 of tracking)
        if len(self.tracked_points) >= 3 and self.release_speed_kmh is None:
            early_speeds = [s for s in self.smoothed_speeds_kmh if s > 40.0]
            if early_speeds:
                self.release_speed_kmh = float(round(float(np.median(early_speeds)), 1))

        if self.last_speed_kmh > self.max_speed_kmh:
            self.max_speed_kmh = self.last_speed_kmh

        return self.last_speed_kmh, self.current_speed_mph

    def record_bounce(self, frame_idx, bounce_info=None):
        """
        Record the ball speed at a pitch bounce event.
        """
        current_kmh = float(self.last_speed_kmh)
        self.bounce_speeds.append({
            "frame": int(frame_idx),
            "speed_kmh": current_kmh,
            "speed_mph": float(round(current_kmh * 0.621371, 1))
        })

    def get_summary(self, include_history=True):
        """
        Compute final delivery speed telemetry metrics.
        """
        active_speeds = [float(s) for s in self.smoothed_speeds_kmh if s >= 35.0]

        # Overall delivery Time-of-Flight cross-check
        time_of_flight_speed_kmh = None
        if len(self.tracked_points) >= 4:
            f_start = self.tracked_points[0][0]
            f_end = self.tracked_points[-1][0]
            total_duration_sec = max(f_end - f_start, 1) / self.fps
            if 0.25 <= total_duration_sec <= 1.2:
                time_of_flight_speed_kmh = float(round((self.pitch_length_m / total_duration_sec) * 3.6, 1))

        if active_speeds:
            avg_kmh = float(np.mean(active_speeds))
            max_kmh = float(np.max(active_speeds))
        elif time_of_flight_speed_kmh:
            avg_kmh = float(time_of_flight_speed_kmh)
            max_kmh = float(time_of_flight_speed_kmh * 1.08)
        else:
            avg_kmh = float(self.last_speed_kmh)
            max_kmh = float(self.max_speed_kmh)

        release_kmh = self.release_speed_kmh
        if release_kmh is None or release_kmh < 40.0:
            if time_of_flight_speed_kmh:
                release_kmh = float(round(time_of_flight_speed_kmh * 1.06, 1))
            elif active_speeds:
                release_kmh = float(round(float(np.percentile(active_speeds, 75)), 1))
            else:
                release_kmh = float(round(avg_kmh, 1))

        self.avg_speed_kmh = float(round(avg_kmh, 1))
        self.max_speed_kmh = float(round(max(max_kmh, release_kmh), 1))
        self.release_speed_kmh = float(round(release_kmh, 1))

        summary = {
            "release_speed_kmh": self.release_speed_kmh,
            "release_speed_mph": float(round(self.release_speed_kmh * 0.621371, 1)),
            "avg_speed_kmh": self.avg_speed_kmh,
            "avg_speed_mph": float(round(self.avg_speed_kmh * 0.621371, 1)),
            "max_speed_kmh": self.max_speed_kmh,
            "max_speed_mph": float(round(self.max_speed_kmh * 0.621371, 1)),
            "bounce_speeds": list(self.bounce_speeds),
            "time_of_flight_speed_kmh": time_of_flight_speed_kmh
        }
        if include_history:
            summary["all_deliveries"] = list(self.all_deliveries)
        return summary

    @staticmethod
    def format_speed_label(speed_kmh, include_mph=True):
        """
        Format speed for video HUD or UI display.
        Example: '138.4 km/h (86.0 mph)'
        """
        if speed_kmh is None or speed_kmh <= 0:
            return "-- km/h"
        if include_mph:
            mph = speed_kmh * 0.621371
            return f"{speed_kmh:.1f} km/h ({mph:.1f} mph)"
        return f"{speed_kmh:.1f} km/h"
