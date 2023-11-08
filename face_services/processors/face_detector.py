from collections import OrderedDict
import logging
from typing import Any, Optional, List
import uuid
from face_services.components.face import Face
from face_services.models.models_list import FACE_DETECTION_MODELS
import numpy as np
import cv2

from face_services.processors.face_helper import resize_frame_dimension

class FaceDetector:
  
	def __init__(self):
		self.id = uuid.uuid4()
		logging.info('FaceDetector {} - Initialize'.format(self.id))
		self.model = None
		self.current_model_name = self.get_available_models()[0]
		self.check_current_model(self.current_model_name)


	@staticmethod
	def get_available_models():
		return list(FACE_DETECTION_MODELS.keys())

	def check_current_model(self, model):
		logging.info('FaceDetector {} - Current model is : {}'.format(self.current_model_name, self.id))
		if model != self.current_model_name and self.model is not None:
			if model is not None and model in self.get_available_models():
				self.current_model_name = model
				logging.info('FaceDetector {} - Initialize with model : {}'.format(self.current_model_name, self.id))
				self.model = cv2.FaceDetectorYN.create(FACE_DETECTION_MODELS[self.current_model_name]['path'], None, (0, 0))
			else:
				logging.info('FaceDetector {} - Model : {} not in {}'.format(self.id, model, self.get_available_models()))	
		elif self.model is None:
			logging.info('FaceDetector {} - Initialize with model : {}'.format(self.id, self.current_model_name))
			self.model = cv2.FaceDetectorYN.create(FACE_DETECTION_MODELS[self.current_model_name]['path'], None, (0, 0))
		else:
			logging.info('FaceDetector {} - Current model is already : {}'.format(self.id, model))	

	def run(self, frame):
		logging.info('FaceDetector {} - Run'.format(self.id))

		faces: List[Face] = []

		# preprocessing
		temp_frame = resize_frame_dimension(frame, 1024, 1024)
		temp_frame_height, temp_frame_width, _ = temp_frame.shape
		frame_height, frame_width, _ = frame.shape
		ratio_height = frame_height / temp_frame_height
		ratio_width = frame_width / temp_frame_width
		self.model.setScoreThreshold(0.5)
		self.model.setNMSThreshold(0.5)
		self.model.setTopK(100)
		self.model.setInputSize((temp_frame_width, temp_frame_height))

		# inference
		_, detections = self.model.detect(temp_frame)

		# postprocessing
		if detections.any():
			for detection in detections:
				bbox =\
				[
					detection[0:4][0] * ratio_width,
					detection[0:4][1] * ratio_height,
					(detection[0:4][0] + detection[0:4][2]) * ratio_width,
					(detection[0:4][1] + detection[0:4][3]) * ratio_height
				]
				face = Face(bbox=bbox, confidence=detection[14])
				face.keypoints = (detection[4:14].reshape((5, 2)) * [[ ratio_width, ratio_height ]]).tolist()
				faces.append(face)
				
		return self.identify_faces(faces)


	def identify_faces(self, detected_faces):
		logging.info('FaceDetector {} - Identify faces'.format(self.id))
		for idx, detected_face in enumerate(detected_faces):
			detected_face.id = idx + 1
		return detected_faces

	@staticmethod
	def get_face_by_id(detected_faces, id):
		for detected_face in detected_faces:
			if detected_face.id == id:
				return detected_face
		return None


