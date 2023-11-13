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


		w2l = W2l(self.source_video.path, self.target_audio.path, 'wav2lip', True, 2, 0, 0, 0,0, None)
		w2l.execute()
		w2luhq = Wav2LipUHQ(self.source_video.path, "GFPGAN", 15, 15, 15, True, None, 2, 0.75, True)
		w2luhq.execute()


		# for frame_idx in tqdm(range(self.source_video.frame_number)):
		# 	current_frame = self.source_video.get_frame_position_by_index(frame_idx)
		# 	cv2.imwrite(os.path.join(self.folder_path, 'frames', str(frame_idx)+'.png'), current_frame)
		# 	detected_faces = self.face_detector.run(current_frame)
		# 	if len(detected_faces) > 0:
		# 		source_face = serialize_faces_analysis(detected_faces)[0]
		# 		with open(os.path.join(self.folder_path, 'faces', str(frame_idx)+'.json'), 'w') as f:
		# 			json.dump(source_face, f)
		# 	else:
		# 		logging.info('VisualDubber - No face detected')
		# 	landmarks = self.face_detector.get_face_3d_features_by_names(detected_faces[0], features_name=['mouth'])
		# 	points = []
		# 	for keypoint in landmarks:
		# 		points.append([int(keypoint[0]), int(keypoint[1])])
		# 	cv2.fillPoly(current_frame, pts=[np.array(points)], color=(255, 0, 0))

		# 	cv2.namedWindow('FaceVisualDubber', 0)
		# 	cv2.imshow('FaceVisualDubber', current_frame)
		# 	cv2.waitKey(1)


		# 	print()

		self.clean_and_close()

		return None
	

	def clean_and_close(self):
		logging.info('VisualDubber - Clean and close')
		if os.path.exists(self.folder_path):
			shutil.rmtree(self.folder_path)

