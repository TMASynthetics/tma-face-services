import logging
from typing import Any, Optional, List
import insightface


class FaceDetector:
  
	def __init__(self):
		logging.info('FaceDetector - Initialize')
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



