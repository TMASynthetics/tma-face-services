from typing import Any, Optional, List
import threading
import insightface
import numpy

from processors.face_analyser import FaceAnalyser


class FaceSwapper:
  
	def __init__(self):
		self.model = insightface.model_zoo.get_model('.assets/models/inswapper_128.onnx', download=False, download_zip=False)
		self.face_analyser = FaceAnalyser()

	def run(self, img_source, img_target):

		face_target = self.face_analyser.run(img_target)
		face_source = self.face_analyser.run(img_source)

		return self.model.get(img_target, face_target[0], face_source[0])












