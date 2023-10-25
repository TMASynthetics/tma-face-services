import logging
from typing import Any, Optional, List
import threading
import insightface
import numpy

from face_services.processors.models import FACE_SWAPPER_MODELS

from ..processors.face_detector import FaceDetector
from ..processors.face_enhancer import FaceEnhancer

class FaceSwapper:
  
	def __init__(self, swapper_model=None, enhancer_model=None):
		logging.info('FaceSwapper - Initialize')
		self.model = None
		self.current_swapper_model_name = self.get_available_models()[0]
		self.check_current_model(swapper_model)
		self.face_analyzer = FaceDetector()
		self.face_enhancer = FaceEnhancer(model=enhancer_model)

	@staticmethod
	def get_available_models():
		return list(FACE_SWAPPER_MODELS.keys())

	def check_current_model(self, model):
		logging.info('FaceSwapper - Current model is : {}'.format(self.current_swapper_model_name))
		if model != self.current_swapper_model_name and self.model is not None:
			if model is not None and model in self.get_available_models():
				self.current_swapper_model_name = model
				logging.info('FaceSwapper - Initialize with model : {}'.format(self.current_swapper_model_name))
				self.model = insightface.model_zoo.get_model(FACE_SWAPPER_MODELS[self.current_swapper_model_name]['path'], download=False, download_zip=False)
			else:
				logging.info('FaceSwapper - Model : {} not in {}'.format(model, self.get_available_models()))	
		elif self.model is None:
			logging.info('FaceSwapper - Initialize with model : {}'.format(self.current_swapper_model_name))
			self.model = insightface.model_zoo.get_model(FACE_SWAPPER_MODELS[self.current_swapper_model_name]['path'], download=False, download_zip=False)
		else:
			logging.info('FaceSwapper - Current model is already : {}'.format(model))	

	def run(self, img_source, img_target, swapper_model=None, enhancer_model=None, enhancer_blend_percentage=None, source_face_id=1, target_face_ids=[], enhance=False):
		logging.info('FaceSwapper - Run')
		swapped_frame = img_target.copy()

		faces_target = self.face_analyzer.run(img_target)
		faces_source = self.face_analyzer.run(img_source)
		self.check_current_model(swapper_model)

		if target_face_ids is None:
			target_face_ids=[]

		if len(faces_source) > 0:
			face_source = FaceDetector.get_face_by_id(faces_source, source_face_id)
			if face_source is not None:
				for face_target in faces_target:
					if face_target.id in target_face_ids or len(target_face_ids)==0 :
						swapped_frame = self.model.get(swapped_frame, face_target, face_source)
			else:
				logging.info('FaceSwapper - No face with id={} in the source image'.format(source_face_id))
		else:
			logging.info('FaceSwapper - No source face detected')

		if enhance:
			logging.info('FaceSwapper - Enhance swapped face(s)')
			swapped_frame = self.face_enhancer.run(swapped_frame, model=enhancer_model, blend_percentage=enhancer_blend_percentage)

		return swapped_frame












