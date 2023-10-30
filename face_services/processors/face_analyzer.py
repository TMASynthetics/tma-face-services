from collections import OrderedDict
import logging
from typing import Any, Optional, List
import insightface


class FaceAnalyzer:
  
	def __init__(self):
		logging.info('FaceAnalyzer - Initialize')
		self.model = insightface.app.FaceAnalysis(name = 'buffalo_l', root='face_services/models/face_detector', providers=['CPUExecutionProvider'])
		self.model.prepare(ctx_id = 0)

	def run(self, frame):
		logging.info('FaceDetector - Run')
		return self.identify_faces(self.model.get(frame))

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

