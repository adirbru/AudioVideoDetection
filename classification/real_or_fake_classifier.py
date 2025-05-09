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
import pathlib


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
                self.video_peaks = [float(entry['timestamp_ms']) for entry in video_data]

            with open(audio_json_path, 'r') as audio_file:
                audio_data = json.load(audio_file)
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
        Match video peaks to audio peaks using closest-time logic within a defined tolerance.

        Returns:
            List[Tuple[float, float]]: List of matched (video_timestamp, audio_timestamp) pairs
        """
        if not self.video_peaks or not self.audio_peaks:
            return []

        all_pairs = []
        for video_time in self.video_peaks:
            for audio_time in self.audio_peaks:
                diff = abs(video_time - audio_time)
                if diff <= self.match_tolerance:
                    all_pairs.append((video_time, audio_time, diff))

        all_pairs.sort(key=lambda x: x[2])

        matched_video = set()
        matched_audio = set()
        matched_pairs = []

        for video_time, audio_time, _ in all_pairs:
            if video_time not in matched_video and audio_time not in matched_audio:
                matched_pairs.append((video_time, audio_time))
                matched_video.add(video_time)
                matched_audio.add(audio_time)

        self.matched_pairs = matched_pairs

        unmatched_video = [v for v in self.video_peaks if v not in matched_video]
        unmatched_audio = [a for a in self.audio_peaks if a not in matched_audio]

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
            is_real = std_diff <= self.threshold_ms

            confidence = min(1.0, abs(std_diff - self.threshold_ms) / self.threshold_ms)
            confidence = round(confidence * 100, 2)

        result = {
            "classification": "real" if is_real else "fake" if is_real is not None else "undetermined",
            "confidence": confidence if is_real is not None else 0.0,
            "metrics": {
                "matched_pairs": len(self.matched_pairs),
                "mean_difference_ms": round(float(np.mean(self.time_differences)), 2),
                "std_difference_ms": round(float(np.std(self.time_differences)), 2) if len(self.time_differences) > 1 else None,
                "min_difference_ms": round(float(np.min(self.time_differences)), 2),
                "max_difference_ms": round(float(np.max(self.time_differences)), 2),
                "range_difference_ms": round(float(np.max(self.time_differences) - np.min(self.time_differences)), 2),
            },
            "threshold_ms": self.threshold_ms
        }

        self.classification_result = result
        return result

    def classify(self, video_json_path: str, audio_json_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Full classification pipeline: load, match, analyze, and optionally save.

        Args:
            video_json_path: Path to video peaks JSON file
            audio_json_path: Path to audio peaks JSON file
            output_path: Optional directory to save results and visualizations

        Returns:
            Classification result dictionary
        """
        self.load_peaks(video_json_path, audio_json_path)
        self.match_peaks()
        self.calculate_time_differences()
        result = self.analyze_differences()
        if not output_path:
            audio_name = pathlib.Path(audio_json_path).stem
            output_path = f"results_{audio_name}"
        self.save_results(output_path)

        classification = result['classification']
        confidence = result['confidence']
        print("\n" + "*" * 30)
        print(f"CLASSIFICATION RESULT: {classification.upper()} (confidence: {confidence}%)")
        print(f"Matched {result['metrics']['matched_pairs']} audio-video peak pairs")
        print(f"Mean time difference: {result['metrics']['mean_difference_ms']} ms")
        print(f"Standard deviation: {result['metrics']['std_difference_ms']} ms")
        print(f"Classification threshold: ±{self.threshold_ms} ms\n")
        return result

    def save_results(self, output_path: str, include_plot: bool = True) -> None:
        """
        Save classification results and matched pairs to CSV/JSON files.

        Args:
            output_path: Directory to save results
            include_plot: Whether to generate and save visualization plots
        """
        os.makedirs(output_path, exist_ok=True)

        if self.matched_pairs and self.time_differences:
            df = pd.DataFrame({
                'video_timestamp': [vt for vt, _ in self.matched_pairs],
                'audio_timestamp': [at for _, at in self.matched_pairs],
                'difference_ms': self.time_differences
            })
            csv_path = os.path.join(output_path, 'matched_peaks.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved matched peaks to {csv_path}")

        if self.classification_result:
            json_path = os.path.join(output_path, 'classification_result.json')
            with open(json_path, 'w') as f:
                json.dump(self.classification_result, f, indent=4)
            print(f"Saved classification results to {json_path}")

        if include_plot and self.matched_pairs and self.time_differences:
            self.plot_time_differences(output_path)
            self.plot_peaks_timeline(output_path)

    def plot_time_differences(self, output_path: str):
        """
        Plot the time difference of each matched pair sequentially.

        Args:
            output_path: Directory to save the plot
        """
        if not self.time_differences:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.time_differences)), self.time_differences, marker='o', linestyle='-')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=self.threshold_ms, color='green', linestyle='--', alpha=0.7, label=f'+{self.threshold_ms}ms threshold')
        plt.axhline(y=-self.threshold_ms, color='green', linestyle='--', alpha=0.7, label=f'-{self.threshold_ms}ms threshold')
        plt.title('Audio-Visual Time Differences Across Matched Pairs')
        plt.xlabel('Matched Pair Index')
        plt.ylabel('Time Difference (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(output_path, 'time_diff_sequence_plot.png')
        plt.savefig(path)
        print(f"Saved time difference sequence plot to {path}")
        plt.close()

    def plot_peaks_timeline(self, output_path: str):
        """
        Plot a timeline showing the alignment of video and audio peaks and their connections.

        Args:
            output_path: Directory to save the plot
        """
        plt.figure(figsize=(12, 6))

        plt.scatter(self.video_peaks, [1] * len(self.video_peaks), color='blue', label='Video peaks', marker='o', s=50, alpha=0.7)
        plt.scatter(self.audio_peaks, [0] * len(self.audio_peaks), color='red', label='Audio peaks', marker='x', s=50, alpha=0.7)

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
        plt.close()


def main():
    """
    Parse command line arguments and run the classifier.
    """
    parser = argparse.ArgumentParser(
        description='Classify videos as real or fake based on audio-visual synchronization')
    parser.add_argument('--video_json', required=True, help='Path to JSON file with video peaks')
    parser.add_argument('--audio_json', required=True, help='Path to JSON file with audio peaks')
    parser.add_argument('--output', default=None, help='Directory to save results (default: results_[audio file name])')
    parser.add_argument('--threshold', type=float, default=100.0, help='Maximum acceptable standard deviation in ms (default: 100.0)')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Maximum time difference to consider peaks as matching in seconds (default: 0.5)')

    args = parser.parse_args()

    if args.output:
        output_dir = args.output
    else:
        audio_name = pathlib.Path(args.audio_json).stem
        output_dir = f"results_{audio_name}"

    classifier = AVClassifier(threshold_ms=args.threshold, match_tolerance=args.tolerance)
    classifier.classify(args.video_json, args.audio_json, output_dir)


if __name__ == "__main__":
    main()
