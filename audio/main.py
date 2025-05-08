import json
import scipy.signal
from pydub import AudioSegment, silence
import wave
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numba as nb
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
import scipy.signal as signal


@nb.njit
def find_peaks(abs_arr, threshold, hold_for_unify):
    starts, ends = [], []
    in_peak = False
    peak_start = 0

    for i, val in enumerate(abs_arr):
        if val >= threshold:
            if not in_peak:
                peak_start = i
                in_peak = True
        else:
            if in_peak:
                peak_end = i - 1
                # Unify with previous peak if close enough
                if starts and (peak_start - ends[-1]) <= hold_for_unify:
                    ends[-1] = peak_end  # Extend previous peak
                else:
                    starts.append(peak_start)
                    ends.append(peak_end)
                in_peak = False

    # Handle case where array ends while in a peak
    if in_peak:
        peak_end = len(abs_arr) - 1
        if starts and (peak_start - ends[-1]) <= hold_for_unify:
            ends[-1] = peak_end
        else:
            starts.append(peak_start)
            ends.append(peak_end)

    return np.array(starts), np.array(ends)


sound_path = r"C:\Users\USER\PycharmProjects\AudioVideoDetection\GalClaps.wav.wav"
sound_path = r"C:\Users\USER\PycharmProjects\AudioVideoDetection\skateboard.wav"
fps = 29.96
minimum_diff_frames = 1
minimum_diff_s = minimum_diff_frames / fps
# Load WAV file
if __name__ == '__main__':
    sound_sample_rate, data = wavfile.read(sound_path)

    # If stereo, convert to mono by averaging channels
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Get absolute values of the audio signal
    abs_samples = np.abs(data)
    threshold = 0.75 * np.max(abs_samples)  # old was max / 2
    plot(abs_samples, sound_sample_rate, threshold=threshold)

    bool_above_threshold = abs_samples > threshold
    indices_above_threshold = np.where(bool_above_threshold)[0]
    diff_of_indices = np.diff(indices_above_threshold)

    valid_diffs = diff_of_indices > minimum_diff_s * sound_sample_rate

    print(f"{len(valid_diffs) = }")

    og_indices = indices_above_threshold[1:][valid_diffs]
    valid_times = og_indices / sound_sample_rate
    print(f"{valid_times = }")

    ind_starts, ind_ends = find_peaks(abs_samples, threshold=threshold, hold_for_unify=minimum_diff_s * sound_sample_rate)
    starts, ends = ind_starts / sound_sample_rate, ind_ends / sound_sample_rate
    for s, e in zip(starts, ends):
        print(s, e)

    plot(abs_samples, sample_rate=sound_sample_rate, threshold=threshold, starts=list(starts), ends=list(ends))

    formatted_times = [{"timestamp_ms": f"{time:.3f}"} for time in starts]

    # Export to JSON
    with open("timestamps.json", "w") as f:
        json.dump(formatted_times, f, indent=2)