import cv2
import numpy as np
import argparse
import time
import json
from pathlib import Path


class SoundActionDetector:
    def __init__(self, sensitivity=0.5):
        self.sensitivity = sensitivity
        self.motion_history = []
        # Significantly increased weights for better detection
        self.vertical_motion_weight = 1.0 + (2.0 * sensitivity)
        self.horizontal_motion_weight = 1.0 + (2.0 * sensitivity)  # Even higher weight for horizontal motion

    def process_video(self, video_path, output_path=None):
        """
        Process video file and detect potential sound-inducing actions

        Args:
            video_path: Path to the input video file
            output_path: Optional path for output visualization video

        Returns:
            List of tuples (timestamp_ms, confidence) where sound actions were detected
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {Path(video_path).name}")
        print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {frame_count}")
        print(f"Processing with sensitivity: {self.sensitivity}")

        # Initialize for motion detection
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame from video")

        # Store several previous frames for more robust detection
        history_length = 4
        frame_history = [None] * history_length

        # Convert and preprocess first frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (15, 15), 0)  # Smaller kernel for finer details

        # Initialize frame history
        for i in range(history_length):
            frame_history[i] = prev_gray.copy()

        # For output visualization
        if output_path:
            print(f"Creating output video at: {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (prev_frame.shape[1], prev_frame.shape[0]))
            if not out.isOpened():
                raise ValueError(f"Could not create output video file: {output_path}")

        # Main detection data
        detected_actions = []
        frame_diffs = []
        frame_number = 0
        frame_data = []  # Store frame data for later visualization

        # Progress tracking
        last_percent = -1
        start_time = time.time()

        # Optical flow parameters - more sensitive to movements like claps
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Initialize points to track
        old_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Show progress
            percent_complete = int((frame_number / frame_count) * 100)
            if percent_complete > last_percent:
                elapsed = time.time() - start_time
                est_total = elapsed / (frame_number / frame_count)
                remain = est_total - elapsed
                print(f"Progress: {percent_complete}% - Est. remaining: {remain:.1f}s", end="\r")
                last_percent = percent_complete

            # Convert to grayscale and blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)

            # Calculate difference from multiple previous frames for better detection
            multi_frame_diff = np.zeros_like(gray)
            for prev in frame_history:
                if prev is not None:
                    frame_diff = cv2.absdiff(prev, gray)
                    multi_frame_diff = cv2.add(multi_frame_diff, frame_diff)

            # Update frame history
            frame_history.pop(0)
            frame_history.append(gray.copy())

            # Threshold with adaptive method
            thresh = cv2.adaptiveThreshold(
                multi_frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Additional preprocessing
            thresh = cv2.medianBlur(thresh, 5)  # Remove noise
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            flow_magnitude = 0
            vertical_flow = 0
            vector_flow = np.array([0, 0])
            horizontal_flow = 0
            # Optical flow for motion direction and speed
            if old_points is not None and len(old_points) > 0:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **lk_params)

                # Filter good points
                if new_points is not None:
                    good_new = new_points[status == 1]
                    good_old = old_points[status == 1]

                    if len(good_new) > 0 and len(good_old) > 0:
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            # Calculate vertical and horizontal components
                            vertical_flow += abs(b - d)  # Vertical movement
                            vector_flow[1] += b-d
                            horizontal_flow += abs(a - c)  # Horizontal movement
                            vector_flow[0] += a-c
                            flow_magnitude += np.sqrt((a - c) ** 2 + (b - d) ** 2)

                            if output_path:
                                # Draw flow arrows with color based on direction
                                color = (0, 255, 0)  # Default green
                                if abs(b - d) > abs(a - c):  # More vertical movement
                                    color = (0, 0, 255)  # Red for vertical
                                elif abs(a - c) > abs(b - d):  # More horizontal movement
                                    color = (255, 0, 0)  # Blue for horizontal
                                cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), color, 2)

                    # Normalize flow components
                    if len(good_new) > 0:
                        flow_magnitude /= len(good_new)
                        vertical_flow /= len(good_new)
                        horizontal_flow /= len(good_new)

            # Find new points to track
            old_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

            # Analyze contour-based motion
            detected_regions = []  # Store regions where sound was detected

            # Calculate combined motion score with directional weights
            combined_score = (flow_magnitude +
                              vertical_flow * self.vertical_motion_weight +
                              horizontal_flow * self.horizontal_motion_weight) / (width * height) * 8000

            frame_diffs.append(combined_score)

            # Store frame data for later visualization
            if output_path:
                frame_data.append({
                    'frame': frame.copy(),
                    'regions': detected_regions.copy() if detected_regions else []
                })

            # Update for next iteration
            prev_gray = gray

            # Update for visualization
            if output_path:
                # Add motion score to visualization
                cv2.putText(frame, f"Motion: {combined_score:.6f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                out.write(frame)

        cap.release()
        if output_path:
            out.release()
            print(f"Output video saved to: {output_path}")

        # Process motion data to detect peaks (sound actions)
        window_size = 5
        smoothed_diffs = np.abs(np.diff(np.convolve(frame_diffs, np.ones(window_size) / window_size, mode='same')))

        # Store full motion history for potential debugging
        self.motion_history = frame_diffs.copy()

        # Calculate adaptive threshold based on video content
        mean_motion = np.mean(smoothed_diffs)
        std_motion = np.std(smoothed_diffs)

        # Adaptive threshold - but ensure we detect something
        adaptive_threshold = min(
            mean_motion + std_motion * (0.8 + self.sensitivity),  # Adaptive component
            np.percentile(smoothed_diffs, 95)  # Fallback to ensure we detect at least some peaks
        )

        # Store for debugging
        self.adaptive_threshold = adaptive_threshold

        print(f"\nAdaptive threshold: {adaptive_threshold:.6f} (mean: {mean_motion:.6f}, std: {std_motion:.6f})")

        # Convert peaks to timestamps and calculate confidence scores
        action_timestamps = []
        if output_path and frame_data:
            print("Creating visualization with detected sound events...")
            # Create new video writer for the visualization
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vis_out = cv2.VideoWriter(output_path, fourcc, fps,
                                      (frame_data[0]['frame'].shape[1], frame_data[0]['frame'].shape[0]))

            for i, frame_info in enumerate(frame_data):
                frame = frame_info['frame']
                # Only draw indicators on peak frames
                if (i < len(frame_data) - 1 # Not reaching the end
                        and min(frame_diffs[i]/frame_diffs[max(i-1, 0)], frame_diffs[max(i+1, 0)]/frame_diffs[max(i-1, 0)]) < 0.5 # Sharp decrease in motion
                        and (not action_timestamps or i - action_timestamps[-1][0] > 5) # Haven't added a similar frame b
                ):
                    confidence = min(1, float(abs(round(1 - frame_diffs[i]/max(frame_diffs[max(i-1, 0)], frame_diffs[max(i-2, 0)]), 3))))

                    # Add timestamp and confidence
                    timestamp_ms = round(i / fps, 3)
                    action_timestamps.append((i, timestamp_ms, confidence))

                    # Draw confidence indicator
                    confidence_color = (
                        int(255 * (1 - confidence)),  # Red component
                        int(255 * confidence),  # Green component
                        0  # Blue component
                    )

                    # Draw confidence bar
                    bar_width = 100
                    bar_height = 20
                    bar_x = 10
                    bar_y = 90

                    # Background bar
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_width, bar_y + bar_height),
                                  (100, 100, 100), -1)

                    # Confidence level
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(bar_width * confidence), bar_y + bar_height),
                                  confidence_color, -1)

                    # Add text
                    cv2.putText(frame, f"Sound Event at {timestamp_ms}ms", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)

                # Add motion score
                cv2.putText(frame, f"Motion: {frame_diffs[i]:.6f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Difference: {frame_diffs[i]-frame_diffs[max(0, i-1)]:.6f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                vis_out.write(frame)

            vis_out.release()
            print(f"Visualization video saved to: {output_path}")

        print(f"\nDetected {len(action_timestamps)} potential sound actions")
        return action_timestamps

    def analyze_actions(self, action_timestamps):
        """
        Classify detected actions (clap, stomp, etc.) - placeholder for future enhancement

        Args:
            action_timestamps: List of tuples (timestamp_ms, confidence) where actions were detected

        Returns:
            Dictionary mapping timestamps to action types
        """
        # This would require a more sophisticated model to implement
        # Currently just returns generic "sound action" for all detections
        return {ts: "sound_action" for ts, _ in action_timestamps}

    def save_results(self, action_timestamps, output_path):
        """
        Save detected actions to a JSON file

        Args:
            action_timestamps: List of tuples (timestamp_ms, confidence) where actions were detected
            output_path: Path to save the results
        """
        results = [
                {
                    "timestamp_ms": ts,
                } for frame, ts, conf in action_timestamps
            ]

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect sound-inducing actions in a silent video')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--output', '-o', help='Path to save detection results JSON')
    parser.add_argument('--visualization', '-v', help='Path to save visualization video')
    parser.add_argument('--sensitivity', '-s', type=float, default=0.7,  # Higher default sensitivity
                        help='Detection sensitivity (0.0-1.0)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Show debug information and save motion graph')

    args = parser.parse_args()

    # Set default output path if not specified
    if not args.output:
        output_path = Path.cwd() / 'video' / Path(args.input_video).with_suffix('.json').name
    else:
        output_path = args.output

    # Initialize and run detector
    detector = SoundActionDetector(sensitivity=args.sensitivity)
    action_timestamps = detector.process_video(args.input_video, Path.cwd() / 'video' / Path(args.input_video).with_suffix('.annotated.mp4').name)
    detector.save_results(action_timestamps, output_path)

    # Debug visualization if requested
    if args.debug and action_timestamps:
        try:
            import matplotlib.pyplot as plt

            # Get video properties
            cap = cv2.VideoCapture(args.input_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Generate time axis in seconds
            time_axis = np.arange(len(detector.motion_history)) / fps

            # Apply same smoothing as in detection
            window_size = max(2, int(fps / 6))
            smoothed_data = np.convolve(detector.motion_history,
                                        np.ones(window_size) / window_size, mode='same')

            # Plot motion data
            plt.figure(figsize=(12, 6))
            plt.plot(time_axis, smoothed_data)

            # Mark detected actions
            action_times = [ts for frame, ts, confidence in action_timestamps]  # Convert ms to seconds
            action_values = [smoothed_data[int(t * fps)] if int(t * fps) < len(smoothed_data) else 0
                             for t in action_times]
            plt.scatter(action_times, action_values, color='red', s=50)

            # Add labels
            plt.xlabel('Time (seconds)')
            plt.ylabel('Motion Magnitude')
            plt.title(f'Motion Analysis: {Path(args.input_video).name}')

            # Add detection threshold
            if hasattr(detector, 'adaptive_threshold'):
                plt.axhline(y=detector.adaptive_threshold, color='r', linestyle='--',
                            label=f'Threshold: {detector.adaptive_threshold:.6f}')

            plt.grid(True)
            plt.legend()

            # Save plot
            plot_path = Path(args.input_video).with_suffix('.motion_analysis.png')
            plt.savefig(str(plot_path))
            print(f"Motion analysis graph saved to {plot_path}")

        except ImportError:
            print("Matplotlib not installed. Skipping debug visualization.")
            print("Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error generating debug visualization: {e}")


if __name__ == "__main__":
    main()
