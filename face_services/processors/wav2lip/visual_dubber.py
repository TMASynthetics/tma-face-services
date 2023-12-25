import time
import numpy as np
import gc
import cv2, os, face_services.processors.wav2lip.audio as audio
from face_services.processors.face_detector import FaceDetector
from face_services.processors.face_enhancer import FaceEnhancer
import subprocess
from tqdm import tqdm
import torch
import face_services.processors.wav2lip.face_detection as face_detection
from face_services.processors.wav2lip.models.wav2lip import Wav2Lip
from pkg_resources import resource_filename
from face_services.logger import logger
from face_services.jobs_database import jobs_database




class VisualDubber:
  
    def __init__(self, video_path, audio_path):

        self.img_size = 96
        self.face = face
        self.audio = audio_path
        self.checkpoint = checkpoint

        self.mel_step_size = 16

        self.batch_size = 128



    def run(self):


        

        pass



    def detect_faces(self):
        pass



    def extract_melspectrogram(self):

        wav = audio.load_wav(self.audio, 16000)
        mel = audio.melspectrogram(wav)

        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        logger.debug('VisualDubber {} - Length of mel chunks: {}'.format(self.id, len(mel_chunks)))