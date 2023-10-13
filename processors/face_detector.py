from typing import Any, Optional, List
import insightface


class FaceDetector:
  
	def __init__(self):
		self.model = insightface.app.FaceAnalysis(name = 'buffalo_l', root='.assets')
		self.model.prepare(ctx_id = 0)

	def run(self, frame):
		return self.identify_faces(self.model.get(frame))

	def identify_faces(self, detected_faces):
		for idx, detected_face in enumerate(detected_faces):
			detected_face.id = idx + 1
		return detected_faces







