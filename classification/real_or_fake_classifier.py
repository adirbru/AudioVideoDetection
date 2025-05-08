#!/usr/bin/env python3
"""
Audio-Visual Classifier for Real/Fake Video Detection

This module analyzes the temporal relationship between audio and video peaks
to determine if a video is real or artificially manipulated.

Usage:
    python classification/real_or_fake_classifier.py --video_json path/to/video_peaks.json --audio_json path/to/audio_peaks.json [options]

Author: Gal Porat (assisted by Claude AI)
Date: May 8, 2025
"""

import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os


class AVClassifier:
    """Classifier that detects if a video is real or fake based on audio-visual synchronization."""

    def __init__(self, threshold_ms: float = 100.0, match_tolerance: float = 0.5):
        """
        Initialize the classifier with configurable parameters.

        Args:
            threshold_ms: Maximum acceptable deviation in milliseconds between audio and video
                         peaks for a video to be classified as real.
            match_tolerance: Maximum time difference (in seconds) to consider an audio and
                            video peak as a matching pair.
        """
        self.threshold_ms = threshold_ms
        self.match_tolerance = match_tolerance
        self.video_peaks = []
        self.audio_peaks = []
        self.matched_pairs = []
        self.time_differences = []
        self.classification_result = {}

    def load_peaks(self, video_json_path: str, audio_json_path: str) -> Tuple[List[float], List[float]]:
        """
        Load timestamp peaks from video and audio JSON files.

        Args:
            video_json_path: Path to the JSON file containing video peak timestamps
            audio_json_path: Path to the JSON file containing audio peak timestamps

        Returns:
            Tuple of (video_peaks, audio_peaks) as lists of timestamps in seconds

        Raises:
            FileNotFoundError: If either JSON file cannot be found
            KeyError: If expected data structure is not found in JSON files
        """
        try:
            with open(video_json_path, 'r') as video_file:
                video_data = json.load(video_file)
                # Extract video timestamps in seconds (already in seconds)
                self.video_peaks = [float(entry['timestamp_ms']) for entry in video_data]

            with open(audio_json_path, 'r') as audio_file:
                audio_data = json.load(audio_file)
                # Extract audio timestamps in seconds (converting from string to float)
                self.audio_peaks = [float(entry['timestamp_ms']) for entry in audio_data]

            print(f"Loaded {len(self.video_peaks)} video peaks and {len(self.audio_peaks)} audio peaks")
            return self.video_peaks, self.audio_peaks

        except FileNotFoundError as e:
            print(f"Error: Could not find JSON file: {e}")
            raise
        except KeyError as e:
            print(f"Error: JSON file is missing expected data structure: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format: {e}")
            raise

    def match_peaks(self) -> List[Tuple[float, float]]:
        """
        Match audio and video peaks based on temporal proximity.
        This version uses a symmetric greedy strategy that finds the best global matches
        within a tolerance window to minimize total time difference.

        Returns:
            List of (video_timestamp, audio_timestamp) matched pairs
        """
        if not self.video_peaks or not self.audio_peaks:
            raise ValueError("Video and audio peaks must be loaded before matching")

        video_peaks = sorted(self.video_peaks)
        audio_peaks = sorted(self.audio_peaks)

        matched_pairs = []
        used_audio = set()

        for v_time in video_peaks:
            # Filter audio peaks not used yet and within tolerance
            candidates = [(i, a_time) for i, a_time in enumerate(audio_peaks)
                          if i not in used_audio and abs(a_time - v_time) <= self.match_tolerance]

            if candidates:
                # Pick the closest audio peak
                i_best, a_best = min(candidates, key=lambda x: abs(x[1] - v_time))
                matched_pairs.append((v_time, a_best))
                used_audio.add(i_best)

        self.matched_pairs = matched_pairs

        unmatched_video = [v for v in video_peaks if v not in [vp for vp, _ in matched_pairs]]
        unmatched_audio = [a for i, a in enumerate(audio_peaks) if i not in used_audio]

        print(f"Matched {len(matched_pairs)} peak pairs")
        print(f"Unmatched: {len(unmatched_video)} video peaks, {len(unmatched_audio)} audio peaks")

        return matched_pairs


    def calculate_time_differences(self) -> List[float]:
        """
        Calculate time differences between matched audio and video peaks in milliseconds.
        Positive values mean audio lags behind video, negative means audio is ahead.

        Returns:
            List of time differences in milliseconds
        """
        if not self.matched_pairs:
            raise ValueError("No matched pairs available. Run match_peaks() first.")

        # Calculate differences: positive if audio comes after video
        time_diffs_ms = [(audio - video) * 1000 for video, audio in self.matched_pairs]
        self.time_differences = time_diffs_ms

        return time_diffs_ms

    def analyze_differences(self) -> Dict:
        """
        Analyze the time differences and determine if the video is real or fake.

        Returns:
            Dictionary with analysis results and classification
        """
        if not self.time_differences:
            raise ValueError("No time differences available. Run calculate_time_differences() first.")

        if len(self.time_differences) < 2:
            print("Warning: Need at least 2 matched pairs for reliable classification")
            is_real = None
            confidence = 0.0
        else:
            mean_diff = np.mean(self.time_differences)
            std_diff = np.std(self.time_differences)
            max_diff = np.max(np.abs(self.time_differences))
            range_diff = np.max(self.time_differences) - np.min(self.time_differences)

            is_real = std_diff <= self.threshold_ms

            if is_real:
                confidence = min(1.0, 1 - (std_diff / self.threshold_ms))
            else:
                confidence = min(1.0, (std_diff - self.threshold_ms) / self.threshold_ms)

            confidence = round(confidence * 100, 2)

        result = {
            "classification": "real" if is_real else "fake" if is_real is not None else "undetermined",
            "confidence": confidence if is_real is not None else 0.0,
            "metrics": {
                "matched_pairs": len(self.matched_pairs),
                "mean_difference_ms": round(float(np.mean(self.time_differences)), 2),
                "std_difference_ms": round(float(np.std(self.time_differences)), 2) if len(
                    self.time_differences) > 1 else None,
                "min_difference_ms": round(float(np.min(self.time_differences)), 2),
                "max_difference_ms": round(float(np.max(self.time_differences)), 2),
                "range_difference_ms": round(float(np.max(self.time_differences) - np.min(self.time_differences)), 2),
            },
            "threshold_ms": self.threshold_ms
        }

        self.classification_result = result
        return result

    def save_results(self, output_path: str, include_plot: bool = True) -> None:
        """
        Save classification results and matched pairs to CSV/JSON files.

        Args:
            output_path: Directory to save results
            include_plot: Whether to generate and save visualization plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Save matched pairs and differences to CSV
        if self.matched_pairs and self.time_differences:
            df = pd.DataFrame({
                'video_timestamp': [vt for vt, _ in self.matched_pairs],
                'audio_timestamp': [at for _, at in self.matched_pairs],
                'difference_ms': self.time_differences
            })
            csv_path = os.path.join(output_path, 'matched_peaks.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved matched peaks to {csv_path}")

        # Save classification results to JSON
        if self.classification_result:
            json_path = os.path.join(output_path, 'classification_result.json')
            with open(json_path, 'w') as f:
                json.dump(self.classification_result, f, indent=4)
            print(f"Saved classification results to {json_path}")

        # Generate and save visualization plots
        if include_plot and self.matched_pairs and self.time_differences:
            self._generate_plots(output_path)

    def _generate_plots(self, output_path: str) -> None:
        """
        Generate visualizations of the audio-visual synchronization analysis.

        Args:
            output_path: Directory to save plots
        """
        # Plot 1: Time differences histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.time_differences, bins=min(20, len(self.time_differences)), alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect sync')
        plt.axvline(x=self.threshold_ms, color='green', linestyle='--', alpha=0.7,
                    label=f'+{self.threshold_ms}ms threshold')
        plt.axvline(x=-self.threshold_ms, color='green', linestyle='--', alpha=0.7,
                    label=f'-{self.threshold_ms}ms threshold')
        plt.title('Distribution of Audio-Visual Time Differences')
        plt.xlabel('Time Difference (ms) [Positive = Audio after Video]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        hist_path = os.path.join(output_path, 'time_diff_histogram.png')
        plt.savefig(hist_path)
        print(f"Saved histogram to {hist_path}")

        # Plot 2: Timeline of peaks
        plt.figure(figsize=(12, 6))

        # Plot all video peaks
        plt.scatter(self.video_peaks, [1] * len(self.video_peaks),
                    color='blue', label='Video peaks', marker='o', s=50, alpha=0.7)

        # Plot all audio peaks
        plt.scatter(self.audio_peaks, [0] * len(self.audio_peaks),
                    color='red', label='Audio peaks', marker='x', s=50, alpha=0.7)

        # Draw lines between matched pairs
        for v_time, a_time in self.matched_pairs:
            plt.plot([v_time, a_time], [1, 0], 'k-', alpha=0.3)

        plt.title('Timeline of Audio and Video Peaks')
        plt.xlabel('Time (seconds)')
        plt.yticks([0, 1], ['Audio', 'Video'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        timeline_path = os.path.join(output_path, 'peaks_timeline.png')
        plt.savefig(timeline_path)
        print(f"Saved timeline plot to {timeline_path}")

        plt.close('all')

    def classify(self, video_json_path: str, audio_json_path: str, output_path: Optional[str] = None,
                 use_confidence_weighting: bool = True) -> Dict:
        """
        Full classification pipeline: load, match, analyze, and optionally save.

        Args:
            video_json_path: Path to video peaks JSON file
            audio_json_path: Path to audio peaks JSON file
            output_path: Optional directory to save results and visualizations
            use_confidence_weighting: If True, weight time differences by video confidence scores

        Returns:
            Classification result dictionary
        """
        self.load_peaks(video_json_path, audio_json_path)
        self.match_peaks()
        self.calculate_time_differences()
        result = self.analyze_differences()

        if output_path:
            self.save_results(output_path)

        classification = result['classification']
        confidence = result['confidence']
        print(f"\nCLASSIFICATION RESULT: {classification.upper()} (confidence: {confidence}%)")
        print(f"Matched {result['metrics']['matched_pairs']} audio-video peak pairs")
        print(f"Mean time difference: {result['metrics']['mean_difference_ms']} ms")
        print(f"Standard deviation: {result['metrics']['std_difference_ms']} ms")
        print(f"Classification threshold: ±{self.threshold_ms} ms\n")

        return result


def main():
    """Parse command line arguments and run the classifier."""
    parser = argparse.ArgumentParser(
        description='Classify videos as real or fake based on audio-visual synchronization')
    parser.add_argument('--video_json', required=True, help='Path to JSON file with video peaks')
    parser.add_argument('--audio_json', required=True, help='Path to JSON file with audio peaks')
    parser.add_argument('--output', default='./results', help='Directory to save results (default: ./results)')
    parser.add_argument('--threshold', type=float, default=100.0,
                        help='Maximum acceptable standard deviation in ms (default: 100.0)')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='Maximum time difference to consider peaks as matching in seconds (default: 0.5)')

    args = parser.parse_args()

    classifier = AVClassifier(threshold_ms=args.threshold, match_tolerance=args.tolerance)
    classifier.classify(args.video_json, args.audio_json, args.output)


if __name__ == "__main__":
    main()
