import logging
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

class Audio:
    def __init__(self, path, sample_rate=None):
        self.path = path
        self.time_series, self.sample_rate = librosa.load(path, sr=sample_rate)
        self.mel = librosa.feature.melspectrogram(y=self.time_series)

    @property    
    def duration(self) -> float:
        return librosa.get_duration(y=self.time_series, sr=self.sample_rate)
    


    # ffmpeg -i tests/files/nwt_40_Mt_F_05.mp3 nwt_40_Mt_F_05.wav