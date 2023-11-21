import json
import logging
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import uuid 
import os

from face_services.components.audio import Audio
from face_services.components.video import Video
from face_services.processors.face_detector import FaceDetector
from face_services.processors.utilities import serialize_faces_analysis
from face_services.processors.wav2lip.w2l import W2l
from face_services.processors.wav2lip.wav2lip_uhq import Wav2LipUHQ

class FaceVisualDubber:
  
	def __init__(self, video_source_path=None, audio_target_path=None):
		logging.info('VisualDubber - Initialize')
		self.source_video = Video(path=video_source_path)
		self.target_audio = Audio(path=audio_target_path)

		self.face_detector = FaceDetector()

		self.mel_step_size = 16
		self.face_detected_batch_size = 16
		self.session_id = 'visualdubber_' + str(uuid.uuid1())

		self.folder_path = os.path.join(os.getcwd(), 'face_services', 'processors', 'temp', self.session_id)
		if not os.path.exists(self.folder_path):
			os.makedirs(self.folder_path)

		for folder in ['frames', 'faces', 'audio', 'output', 'debug']:
			if not os.path.exists(os.path.join(self.folder_path, folder)):
				os.makedirs(os.path.join(self.folder_path, folder))
	


	def run(self, video_source_path=None, audio_target_path=None):
		logging.info('VisualDubber - Run')


		w2l = W2l(self.source_video.path, self.target_audio.path, 'wav2lip', True, 2, 0, 20, 0, 0, None, self.folder_path)
		w2l.execute()
		# w2luhq = Wav2LipUHQ(self.source_video.path, "GFPGAN", 15, 15, 15, True, None, 2, 80, self.folder_path, False)
		# w2luhq.execute()
		
		return self.clean_and_close()
	

	def clean_and_close(self):
		logging.info('VisualDubber - Clean and close')
		shutil.move(os.path.join(self.folder_path, "output", "result_voice.mp4"), self.session_id + '.mp4')
		if os.path.exists(self.folder_path):
			shutil.rmtree(self.folder_path)
		return self.session_id + '.mp4'

