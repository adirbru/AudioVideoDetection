from pathlib import Path
from audio.main import audio_peaks_detection
from video.main import video_peaks_detection
from classification.real_or_fake_classifier import AVClassifier

# video_name = "GalClaps.mp4" # Real
# video_name = "droppingDucks2.mp4" # Real
# video_name = "clap_with_sound.mp4" # Fake
video_name = "skateboard.mp4" # Fake
audio_peaks_detection(video_name)
video_peaks_detection(video_name)
json_path = Path(video_name).with_suffix('.json')
classifier = AVClassifier(threshold_ms=80.0, match_tolerance=0.5)
classifier.classify('video' / json_path, 'audio' / json_path)
