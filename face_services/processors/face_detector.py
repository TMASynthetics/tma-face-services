from collections import OrderedDict
import logging
from typing import Any, Optional, List
import insightface
import onnxruntime
from face_services.components.face import Face
from face_services.models.models_list import FACE_DETECTION_MODELS
import numpy as np
import cv2
Frame = np.ndarray[Any, Any]

def resize_frame_dimension(frame : Frame, max_width : int, max_height : int) -> Frame:
	height, width = frame.shape[:2]
	if height > max_height or width > max_width:
		scale = min(max_height / height, max_width / width)
		new_width = int(width * scale)
		new_height = int(height * scale)
		return cv2.resize(frame, (new_width, new_height))
	return frame

class FaceDetector:
  
	def __init__(self):
		logging.info('FaceDetector - Initialize')
		self.model = None
		self.current_model_name = self.get_available_models()[0]
		self.check_current_model(self.current_model_name)


	@staticmethod
	def get_available_models():
		return list(FACE_DETECTION_MODELS.keys())

	def check_current_model(self, model):
		logging.info('FaceDetector - Current model is : {}'.format(self.current_model_name))
		if model != self.current_model_name and self.model is not None:
			if model is not None and model in self.get_available_models():
				self.current_model_name = model
				logging.info('FaceDetector - Initialize with model : {}'.format(self.current_model_name))
				self.model = cv2.FaceDetectorYN.create(FACE_DETECTION_MODELS[self.current_model_name]['path'], None, (0, 0))
			else:
				logging.info('FaceDetector - Model : {} not in {}'.format(model, self.get_available_models()))	
		elif self.model is None:
			logging.info('FaceDetector - Initialize with model : {}'.format(self.current_model_name))
			self.model = cv2.FaceDetectorYN.create(FACE_DETECTION_MODELS[self.current_model_name]['path'], None, (0, 0))
		else:
			logging.info('FaceDetector - Current model is already : {}'.format(model))	

	def run(self, frame):
		logging.info('FaceDetector - Run')

		faces: List[Face] = []

		temp_frame = resize_frame_dimension(frame, 1024, 1024)
		temp_frame_height, temp_frame_width, _ = temp_frame.shape
		frame_height, frame_width, _ = frame.shape
		ratio_height = frame_height / temp_frame_height
		ratio_width = frame_width / temp_frame_width
		self.model.setScoreThreshold(0.5)
		self.model.setNMSThreshold(0.5)
		self.model.setTopK(100)
		self.model.setInputSize((temp_frame_width, temp_frame_height))


		_, detections = self.model.detect(temp_frame)
		if detections.any():
			for detection in detections:
				bbox =\
				[
					detection[0:4][0] * ratio_width,
					detection[0:4][1] * ratio_height,
					(detection[0:4][0] + detection[0:4][2]) * ratio_width,
					(detection[0:4][1] + detection[0:4][3]) * ratio_height
				]
				kps = (detection[4:14].reshape((5, 2)) * [[ ratio_width, ratio_height ]])
				score = detection[14]
				face = Face(bbox=bbox, confidence=score)
				face.landmarks_2d_5 = kps.tolist()
				faces.append(face)



		return self.identify_faces(faces)


	def identify_faces(self, detected_faces):
		logging.info('FaceDetector - Identify faces')
		for idx, detected_face in enumerate(detected_faces):
			detected_face.id = idx + 1
		return detected_faces

	@staticmethod
	def get_face_by_id(detected_faces, id):
		for detected_face in detected_faces:
			if detected_face.id == id:
				return detected_face
		return None

	@staticmethod
	def get_face_3d_features_by_names(detected_face, features_name=[]):
		facial_features = []
		for feature_name in features_name:
			if feature_name in FACIAL_LANDMARKS_IDXS.keys():
				facial_features += detected_face.landmark_3d_68[FACIAL_LANDMARKS_IDXS[feature_name][0]:FACIAL_LANDMARKS_IDXS[feature_name][-1]]
		return facial_features

