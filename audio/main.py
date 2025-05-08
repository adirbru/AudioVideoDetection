from pathlib import Path

import numpy as np
import numba as nb
import json

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.io import wavfile
import os

def plot(abs_samples, sample_rate, threshold=0, starts=[], ends=[]):
    time_axis = np.linspace(0, len(abs_samples) / sample_rate, num=len(abs_samples))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, abs_samples, color='blue')
    plt.title('Absolute Audio Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (abs)')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold}')

    for start in starts:
        plt.axvline(x=start, color='green', linestyle='--', label='Start' if starts.index(start) == 0 else "")
    for end in ends:
        plt.axvline(x=end, color='red', linestyle='--', label='End' if ends.index(end) == 0 else "")

    plt.grid(True)
    plt.tight_layout()
    plt.show()



@nb.njit
def find_peaks(abs_arr, threshold, hold_for_unify):
    starts, ends = [], []
    in_peak = False
    peak_start = 0
    max_in_peak = 0
    max_ind_in_peak = 0
    max_indices = []

    for i, val in enumerate(abs_arr):
        if val >= threshold:
            if not in_peak:
                peak_start = i
                in_peak = True
            if val >= max_in_peak:
                max_in_peak = val
                max_ind_in_peak = i
        else:
            if in_peak:
                peak_end = i - 1
                if starts and (peak_start - ends[-1]) <= hold_for_unify:
                    ends[-1] = peak_end
                    if max_in_peak > abs_arr[max_indices[-1]]:
                        max_indices[-1] = max_ind_in_peak
                else:
                    starts.append(peak_start)
                    ends.append(peak_end)
                    max_indices.append(max_ind_in_peak)
                max_in_peak = 0
                in_peak = False

    # Handle case where array ends while in a peak
    if in_peak:
        peak_end = len(abs_arr) - 1
        if starts and (peak_start - ends[-1]) <= hold_for_unify:
            ends[-1] = peak_end
            if max_in_peak > abs_arr[max_indices[-1]]:
                max_indices[-1] = max_ind_in_peak
        else:
            starts.append(peak_start)
            ends.append(peak_end)
            max_indices.append(max_ind_in_peak)

    return np.array(starts), np.array(ends), np.array(max_indices)
def get_abs_sound(sound_path):
    sound_sample_rate, data = wavfile.read(sound_path)
    # If stereo, convert to mono by averaging channels
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    # Get absolute values of the audio signal
    abs_samples = np.abs(data)
    return abs_samples, sound_sample_rate

def write_to_json(starts, name):
    formatted_times = [{"timestamp_ms": f"{time:.3f}"} for time in starts]
    with open(f"{name}.json", "w") as f:
        json.dump(formatted_times, f, indent=2)

name = ["GalClaps", "skateboard", "clap_with_sound"][2]

video_path = str(Path.cwd() / ".." /f"{name}.mp4")
sound_path = f"{video_path[:-4]}.wav"
os.system(f'ffmpeg -y -i {video_path} {sound_path}')

fps = 24
minimum_diff_frames = 1
minimum_diff_s = minimum_diff_frames / fps
FACTOR_TOO_BIG = 2
# Load WAV file
write = True
def main():
    abs_samples, sound_sample_rate = get_abs_sound(sound_path)
    # threshold = 0.75 * np.max(abs_samples) if skate else 0.5 * np.max(abs_samples)
    last_peaks_num = 0
    last_threshold = np.max(abs_samples)
    for i, threshold in enumerate(np.arange(np.max(abs_samples), np.max(abs_samples) / 10, -np.max(abs_samples) / 10)):
        starts, ends = get_starts_and_ends(abs_samples, sound_sample_rate, threshold)
        plot(abs_samples, sample_rate=sound_sample_rate, threshold=threshold, starts=list(starts), ends=list(ends))
        current_peaks_num = len(starts)

        if current_peaks_num < last_peaks_num:
            threshold = last_threshold
            break
        elif current_peaks_num <= 10:
            pass
        elif current_peaks_num > last_peaks_num * 5:
            threshold = last_threshold
            break

        if np.max(ends - starts) > minimum_diff_s * FACTOR_TOO_BIG:
            threshold = last_threshold
            break

        last_peaks_num = current_peaks_num
        last_threshold = threshold

    starts, ends = get_starts_and_ends(abs_samples, sound_sample_rate, threshold)
    plot(abs_samples, sample_rate=sound_sample_rate, threshold=threshold, starts=list(starts), ends=list(ends))
    print(f"{i = }")
    if write:
        write_to_json(starts, name)


def get_starts_and_ends(abs_samples, sound_sample_rate, threshold):
    ind_starts, ind_ends, _ = find_peaks(abs_samples, threshold=threshold,
                                         hold_for_unify=minimum_diff_s * sound_sample_rate)
    starts, ends = ind_starts / sound_sample_rate, ind_ends / sound_sample_rate
    for s, e in zip(starts, ends):
        print(s, e)
    print()
    return starts, ends


if __name__ == '__main__':
    main()
