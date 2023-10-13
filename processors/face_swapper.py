from typing import Any, Optional, List
import threading
import insightface
import numpy

from processors.face_detector import FaceDetector
from processors.face_enhancer import FaceEnhancer

class FaceSwapper:
  
	def __init__(self):
		self.model = insightface.model_zoo.get_model('.assets/models/inswapper_128.onnx', download=False, download_zip=False)
		self.face_analyzer = FaceDetector()
		self.face_enhancer = FaceEnhancer()

	def run(self, img_source, img_target, enhance=False):

		swapped_frame = img_target.copy()

		faces_target = self.face_analyzer.run(img_target)
		faces_source = self.face_analyzer.run(img_source)

		if len(faces_source) > 0:
			for face_target in faces_target:
				swapped_frame = self.model.get(swapped_frame, face_target, faces_source[0])

		if enhance:
			swapped_frame = self.face_enhancer.run(swapped_frame)

		return swapped_frame












