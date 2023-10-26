import json
import logging
import shutil
import cv2
from tqdm import tqdm
import uuid 
import os

from face_services.components.audio import Audio
from face_services.components.video import Video
from face_services.processors.face_detector import FaceDetector
from face_services.processors.utilities import serialize_faces_analysis

class FaceVisualDubber:
  
	def __init__(self, video_source_path, audio_target_path):
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


		for folder in ['frames', 'faces']:
			if not os.path.exists(os.path.join(self.folder_path, folder)):
				os.makedirs(os.path.join(self.folder_path, folder))
	


	def run(self):
		logging.info('VisualDubber - Run')

		for frame_idx in tqdm(range(self.source_video.frame_number)):

			current_frame = self.source_video.get_frame_position_by_index(frame_idx)
			cv2.imwrite(os.path.join(self.folder_path, 'frames', str(frame_idx)+'.png'), current_frame)


			detected_faces = self.face_detector.run(current_frame)
			if len(detected_faces) > 0:
				source_face = serialize_faces_analysis(detected_faces)[0]
				with open(os.path.join(self.folder_path, 'faces', str(frame_idx)+'.json'), 'w') as f:
					json.dump(source_face, f)
			else:
				logging.info('VisualDubber - No face detected')









			cv2.namedWindow('FaceVisualDubber', 0)
			cv2.imshow('FaceVisualDubber', current_frame)
			cv2.waitKey(1)


			print()

		self.clean_and_close()

		return None
	

	def clean_and_close(self):
		logging.info('VisualDubber - Clean and close')
		if os.path.exists(self.folder_path):
			shutil.rmtree(self.folder_path)

