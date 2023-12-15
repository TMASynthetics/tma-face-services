import json
from face_services.logger import logger
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import uuid 
import os
from face_services.components.audio import Audio
from face_services.components.video import Video
from face_services.processors.face_detector import FaceDetector
from face_services.processors.wav2lip.w2l import W2l
from face_services.processors.wav2lip.wav2lip_uhq import Wav2LipUHQ
from face_services.jobs_database import jobs_database

class FaceVisualDubber:
  
	def __init__(self, video_source_path=None, audio_target_path=None):
		self.id = str(uuid.uuid1())
		logger.info('VisualDubber {} - Initialize'.format(self.id))
		if video_source_path:
			self.source_video = Video(path=video_source_path)
		if audio_target_path:
			self.target_audio = Audio(path=audio_target_path)

		self.face_detector = FaceDetector()

		self.mel_step_size = 16
		self.face_detected_batch_size = 16

		self.folder_path = os.path.join(os.getcwd(), 'face_services', 'processors', 'temp', self.id)
		if not os.path.exists(self.folder_path):
			os.makedirs(self.folder_path)

		for folder in ['frames', 'faces', 'audio', 'output', 'debug', 'faces_enhanced', 'frames_processed']:
			if not os.path.exists(os.path.join(self.folder_path, folder)):
				os.makedirs(os.path.join(self.folder_path, folder))
	
	def run(self, model=None):
		logger.info('VisualDubber {} - Run'.format(self.id))

		if model is None or model not in self.get_available_models():
			model = self.get_available_models()[0]
		logger.info('VisualDubber {} - Current model is : {}'.format(self.id, model))

		w2l = W2l(self.source_video.path, 
			self.target_audio.path, 
			model, True, 1, 0, 20, 0, 0, None, 
			self.folder_path, self.id)
		
		w2l_output = w2l.execute()

		w2luhq = Wav2LipUHQ(self.source_video.path, w2l_output, "GFPGAN", 30, 15, 15, True, None, 
					  1, 75, self.folder_path, False, self.target_audio.path)
		w2luhq.execute()

		Video.create_video_from_images(os.path.join(self.folder_path, 'frames_processed'), 
								 os.path.join(self.folder_path, 'output', 'result_enhanced.mp4'),
								 self.source_video.fps, self.target_audio.path)

		output_path = self.clean_and_close()

		if self.id in jobs_database.keys():
			jobs_database[self.id]['progress'] = 1
			jobs_database[self.id]['path'] = output_path

		return output_path

	def clean_and_close(self):
		logger.info('VisualDubber {} - Clean and close'.format(self.id))
		if not os.path.exists('outputs'):
			os.makedirs('outputs')
		shutil.move(os.path.join(self.folder_path, "output", "result_voice.mp4"), 
			  		os.path.join('outputs', self.id + '.mp4'))
		shutil.move(os.path.join(self.folder_path, "output", "result_enhanced.mp4"), 
			  		os.path.join('outputs', self.id + '_enhanced.mp4'))
		if os.path.exists(self.folder_path):
			shutil.rmtree(self.folder_path)
		return os.path.join('outputs', self.id + '_enhanced.mp4')

	@staticmethod
	def get_available_models():
		return ['wav2lip', 'wav2lip_gan']