import logging
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
from face_services.processors.visual_dubbing.hparams import hparams as hp

class Audio:
    def __init__(self, path, sample_rate=None):
        self.path = path

        # self.time_series, self.sample_rate = librosa.load(path, sr=sample_rate)  #, offset=0, duration=0)
        # self.mel = librosa.feature.melspectrogram(y=self.time_series)

        self.mel_chunks = []
        self.current_mel_chunk = []
        self._mel_basis = None

        self.time_series, self.sample_rate = librosa.core.load(path, sr=sample_rate)
        self.mel = self.melspectrogram()

    def duration(self) -> float:
        return librosa.get_duration(y=self.time_series, sr=self.sample_rate)
    
    @property    
    def name(self) -> str:
        return self.path.split('/')[-1].split('.')[0]

    def melspectrogram(self):
        D = Audio._stft(Audio.preemphasis(self.time_series, hp.preemphasis, hp.preemphasize))
        S = Audio._amp_to_db(self._linear_to_mel(np.abs(D))) - hp.ref_level_db

        if hp.signal_normalization:
            return Audio._normalize(S)
        return S

    @staticmethod
    def _stft(y):
        if hp.use_lws:
            return Audio._lws_processor(hp).stft(y).T
        else:
            return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=Audio.get_hop_size(), win_length=hp.win_size)


    @staticmethod
    def preemphasis(wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    @staticmethod
    def inv_preemphasis(wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    @staticmethod
    def get_hop_size():
        hop_size = hp.hop_size
        if hop_size is None:
            assert hp.frame_shift_ms is not None
            hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
        return hop_size

    @staticmethod
    def linearspectrogram(wav):
        D = Audio._stft(Audio.preemphasis(wav, hp.preemphasis, hp.preemphasize))
        S = Audio._amp_to_db(np.abs(D)) - hp.ref_level_db

        if hp.signal_normalization:
            return Audio._normalize(S)
        return S
    

    def _linear_to_mel(self, spectogram):
        if self._mel_basis is None:
            self._mel_basis = Audio._build_mel_basis()
        return np.dot(self._mel_basis, spectogram)
    
    @staticmethod
    def librosa_pad_lr(x, fsize, fshift): # Librosa correct padding
        return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

    @staticmethod
    def _build_mel_basis():
        assert hp.fmax <= hp.sample_rate // 2
        return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                                fmin=hp.fmin, fmax=hp.fmax)

    @staticmethod
    def _amp_to_db(x):
        min_level = np.exp(hp.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, (x) * 0.05)

    @staticmethod
    def _normalize(S):
        if hp.allow_clipping_in_normalization:
            if hp.symmetric_mels:
                return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                            -hp.max_abs_value, hp.max_abs_value)
            else:
                return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

        assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
        if hp.symmetric_mels:
            return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
        else:
            return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

    @staticmethod
    def _denormalize(D):
        if hp.allow_clipping_in_normalization:
            if hp.symmetric_mels:
                return (((np.clip(D, -hp.max_abs_value,
                                hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                        + hp.min_level_db)
            else:
                return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

        if hp.symmetric_mels:
            return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
        else:
            return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
