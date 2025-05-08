import wave
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def plot(abs_samples, sample_rate, threshold=0):
    time_axis = np.linspace(0, len(abs_samples) / sample_rate, num=len(abs_samples))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, abs_samples, color='blue')
    plt.title('Absolute Audio Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (abs)')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold}')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

sound_path = r"C:\Users\USER\PycharmProjects\AudioVideoDetection\GalClaps.wav.wav"
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
    threshold = np.max(abs_samples) / 2
    plot(abs_samples, sound_sample_rate, threshold=threshold)

    bool_above_threshold = abs_samples > threshold
    indices_above_threshold = np.where(bool_above_threshold)[0]
    diff_of_indices = np.diff(indices_above_threshold)

    valid_diffs = diff_of_indices > minimum_diff_s * sound_sample_rate

    print(f"{len(valid_diffs) = }")

    og_indices = indices_above_threshold[1:][valid_diffs]
    valid_times = og_indices / sound_sample_rate
    print(valid_times)
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_wav("example.wav")
    chunks = silence.detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)



    for start, end in chunks:
        print(f"Sound from {start / 1000:.2f}s to {end / 1000:.2f}s")
