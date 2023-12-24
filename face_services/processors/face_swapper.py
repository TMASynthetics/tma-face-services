from face_services.components.video import Video
from face_services.logger import logger
from typing import Any, Optional, List
import threading
import uuid
import insightface
import numpy
from onnx import numpy_helper
import onnx
import onnxruntime
from face_services.processors.utilities import onnx_providers
from face_services.components.face import Face

from face_services.models_list import FACE_SWAPPER_MODELS
from face_services.processors.face_helper import paste_back, warp_face
from face_services.typing import Embedding, Frame

from .face_detector import FaceDetector
from ..processors.face_enhancer import FaceEnhancer

class FaceSwapper:
  
	def __init__(self, swapper_model=None, enhancer_model=None):
		self.id = str(uuid.uuid4())
		logger.debug('FaceSwapper {} - Initialize'.format(self.id))
		self.model = None
		self.current_swapper_model_name = self.get_available_models()[0]
		self.check_current_model(swapper_model)
		self.face_detector = FaceDetector()
		self.face_enhancer = FaceEnhancer(model=enhancer_model)


	def set_source_face(self, img_source):
		self.faces_source = self.face_detector.run(img_source)

	@staticmethod
	def get_available_models():
		return list(FACE_SWAPPER_MODELS.keys())

	def check_current_model(self, model):
		logger.debug('FaceSwapper - Current model is : {}'.format(self.current_swapper_model_name))
		if model != self.current_swapper_model_name and self.model is not None and model is not None:
			if model is not None and model in self.get_available_models():
				self.current_swapper_model_name = model
				logger.debug('FaceSwapper - Initialize with model : {}'.format(self.current_swapper_model_name))
				self.model = onnxruntime.InferenceSession(FACE_SWAPPER_MODELS[self.current_swapper_model_name]['path'], providers = onnx_providers)
			else:
				logger.debug('FaceSwapper - Model : {} not in {}'.format(model, self.get_available_models()))	
		elif self.model is None:
			logger.debug('FaceSwapper - Initialize with model : {}'.format(self.current_swapper_model_name))
			self.model = onnxruntime.InferenceSession(FACE_SWAPPER_MODELS[self.current_swapper_model_name]['path'], providers = onnx_providers)

	def run(self, img_target: Frame, 
		 img_source: Frame = None, 
		 swapper_model=None, enhancer_model=None, 
		 enhancer_blend_percentage=80, 
		 source_face_id=1, 
		 target_face_ids=[], 
		 enhance=False):
		
		logger.debug('FaceSwapper - Run')
		swapped_frame = img_target.copy()

		faces_target = self.face_detector.run(img_target)
		if img_source is not None:
			self.faces_source = self.face_detector.run(img_source)

		self.check_current_model(swapper_model)

		if target_face_ids is None:
			target_face_ids=[]

		if len(self.faces_source) > 0:
			face_source = FaceDetector.get_face_by_id(self.faces_source, source_face_id)
			if face_source is not None:
				for face_target in faces_target:
					if face_target.id in target_face_ids or len(target_face_ids)==0 :
						
						

						model_size = FACE_SWAPPER_MODELS[self.current_swapper_model_name]['size']
						model_template = FACE_SWAPPER_MODELS[self.current_swapper_model_name]['template']
						crop_frame, affine_matrix = warp_face(swapped_frame, face_target.keypoints, model_template, model_size)
						crop_frame = self.prepare_crop_frame(crop_frame)
						frame_processor_inputs = {}
						for frame_processor_input in self.model.get_inputs():
							if frame_processor_input.name == 'source':
								frame_processor_inputs[frame_processor_input.name] = self.prepare_source_face(face_source)
							if frame_processor_input.name == 'source_embedding':
								frame_processor_inputs[frame_processor_input.name] = self.prepare_source_embedding(source_face) # type: ignore[assignment]
							if frame_processor_input.name == 'target':
								frame_processor_inputs[frame_processor_input.name] = crop_frame # type: ignore[assignment]

						output = self.model.run(None, frame_processor_inputs)[0][0]

						crop_frame = self.normalize_crop_frame(output)
						swapped_frame = paste_back(swapped_frame, crop_frame, affine_matrix)




			
			else:
				logger.debug('FaceSwapper - No face with id={} in the source image'.format(source_face_id))
		else:
			logger.debug('FaceSwapper - No source face detected')

		if enhance:
			logger.debug('FaceSwapper - Enhance swapped face(s)')
			swapped_frame = self.face_enhancer.run(swapped_frame, model=enhancer_model, blend_percentage=enhancer_blend_percentage)

		return swapped_frame






	def get_model_matrix(self) -> Any:
		model_path = FACE_SWAPPER_MODELS[self.get_available_models()[0]]['path']
		model = onnx.load(model_path)
		return numpy_helper.to_array(model.graph.initializer[-1])



	def prepare_source_face(self, source_face : Face) -> Face:
		model_matrix = self.get_model_matrix()
		source_face = source_face.embedding.reshape((1, -1))
		source_face = numpy.dot(source_face, model_matrix) / numpy.linalg.norm(source_face)
		return source_face


	def prepare_source_embedding(self, source_face : Face) -> Embedding:
		source_embedding = source_face.normed_embedding.reshape(1, -1)
		return source_embedding


	def prepare_crop_frame(self, crop_frame : Frame) -> Frame:
		crop_frame = crop_frame / 255.0
		crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
		crop_frame = numpy.expand_dims(crop_frame, axis = 0).astype(numpy.float32)
		return crop_frame


	def normalize_crop_frame(self, crop_frame : Frame) -> Frame:
		crop_frame = crop_frame.transpose(1, 2, 0)
		crop_frame = (crop_frame * 255.0).round()
		crop_frame = crop_frame[:, :, ::-1].astype(numpy.uint8)
		return crop_frame



