import logging
from typing import Any, Optional, List
import threading
import insightface
import numpy

from ..processors.face_detector import FaceDetector
from ..processors.face_enhancer import FaceEnhancer

class FaceSwapper:
  
	def __init__(self):
		logging.info('FaceSwapper - Initialize')
		self.model = insightface.model_zoo.get_model('face_services/.assets/models/inswapper_128.onnx', download=False, download_zip=False)
		self.face_analyzer = FaceDetector()
		self.face_enhancer = FaceEnhancer()
		
	def run(self, img_source, img_target, source_face_id=1, target_face_ids=[], enhance=False):
		logging.info('FaceSwapper - Run')
		swapped_frame = img_target.copy()

		faces_target = self.face_analyzer.run(img_target)
		faces_source = self.face_analyzer.run(img_source)

		if target_face_ids is None:
			target_face_ids=[]

		face_source = FaceDetector.get_face_by_id(faces_source, source_face_id)

		if face_source is not None:
			for face_target in faces_target:
				if face_target.id in target_face_ids or len(target_face_ids)==0 :
					swapped_frame = self.model.get(swapped_frame, face_target, face_source)

		if enhance:
			swapped_frame = self.face_enhancer.run(swapped_frame)

		return swapped_frame












