from typing import Any, Dict, Optional, List, Tuple
import uuid
import numpy as np
from face_services.models.models_list import FACE_ENHANCER_MODELS
from .face_detector import FaceDetector
import logging
import onnxruntime
import cv2


class FaceEnhancer:
  
	def __init__(self, model=None):
		self.id = uuid.uuid4()
		logging.info('FaceEnhancer {} - Initialize'.format(self.id))
		self.model = None
		self.current_model_name = self.get_available_models()[0]
		self.check_current_model(model)
		self.face_detector = FaceDetector()

	@staticmethod
	def get_available_models():
		return list(FACE_ENHANCER_MODELS.keys())

	def check_current_model(self, model):
		logging.info('FaceEnhancer - Current model is : {}'.format(self.current_model_name))
		if model != self.current_model_name and self.model is not None:
			if model is not None and model in self.get_available_models():
				self.current_model_name = model
				logging.info('FaceEnhancer - Initialize with model : {}'.format(self.current_model_name))
				self.model = onnxruntime.InferenceSession(FACE_ENHANCER_MODELS[self.current_model_name]['path'], providers = ['CPUExecutionProvider'])
			else:
				logging.info('FaceEnhancer - Model : {} not in {}'.format(model, self.get_available_models()))
		elif self.model is None:
			logging.info('FaceEnhancer - Initialize with model : {}'.format(self.current_model_name))
			self.model = onnxruntime.InferenceSession(FACE_ENHANCER_MODELS[self.current_model_name]['path'], providers = ['CPUExecutionProvider'])
		else:
			logging.info('FaceEnhancer - Current model is already : {}'.format(model))	
		
	def run(self, frame, model: str=None, blend_percentage: int=100):
		logging.info('FaceEnhancer - Run with blend_percentage : {}%'.format(blend_percentage))
		
		enhanced_frame = frame.copy()
		analysed_faces = self.face_detector.run(frame)
		self.check_current_model(model)

		for face in analysed_faces:
			crop_frame, affine_matrix = self.warp_face(face, enhanced_frame)
			crop_frame = self.prepare_crop_frame(crop_frame)
			frame_processor_inputs = {}
			for frame_processor_input in self.model.get_inputs():
				if frame_processor_input.name == 'input':
					frame_processor_inputs[frame_processor_input.name] = crop_frame
				if frame_processor_input.name == 'weight':
					frame_processor_inputs[frame_processor_input.name] = np.array([ 1 ], dtype = np.double)
			crop_frame = self.model.run(None, frame_processor_inputs)[0][0]
			crop_frame = self.normalize_crop_frame(crop_frame)
			paste_frame = self.paste_back(enhanced_frame, crop_frame, affine_matrix)
			enhanced_frame = self.blend_frame(enhanced_frame, paste_frame, blend_percentage)

		return enhanced_frame



	def warp_face(self, target_face, temp_frame):
		template = np.array(
		[
			[ 192.98138, 239.94708 ],
			[ 318.90277, 240.1936 ],
			[ 256.63416, 314.01935 ],
			[ 201.26117, 371.41043 ],
			[ 313.08905, 371.15118 ]
		])
		affine_matrix = cv2.estimateAffinePartial2D(np.array(target_face.keypoints), template, method = cv2.LMEDS)[0]
		crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (512, 512))
		return crop_frame, affine_matrix


	def prepare_crop_frame(self, crop_frame):
		crop_frame = crop_frame[:, :, ::-1] / 255.0
		crop_frame = (crop_frame - 0.5) / 0.5
		crop_frame = np.expand_dims(crop_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
		return crop_frame


	def normalize_crop_frame(self, crop_frame):
		crop_frame = np.clip(crop_frame, -1, 1)
		crop_frame = (crop_frame + 1) / 2
		crop_frame = crop_frame.transpose(1, 2, 0)
		crop_frame = (crop_frame * 255.0).round()
		crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
		return crop_frame


	def paste_back(self, temp_frame, crop_frame, affine_matrix):
		inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
		temp_frame_height, temp_frame_width = temp_frame.shape[0:2]
		crop_frame_height, crop_frame_width = crop_frame.shape[0:2]
		inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
		inverse_mask = np.ones((crop_frame_height, crop_frame_width, 3), dtype = np.float32)
		inverse_mask_frame = cv2.warpAffine(inverse_mask, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
		inverse_mask_frame = cv2.erode(inverse_mask_frame, np.ones((2, 2)))
		inverse_mask_border = inverse_mask_frame * inverse_crop_frame
		inverse_mask_area = np.sum(inverse_mask_frame) // 3
		inverse_mask_edge = int(inverse_mask_area ** 0.5) // 20
		inverse_mask_radius = inverse_mask_edge * 2
		inverse_mask_center = cv2.erode(inverse_mask_frame, np.ones((inverse_mask_radius, inverse_mask_radius)))
		inverse_mask_blur_size = inverse_mask_edge * 2 + 1
		inverse_mask_blur_area = cv2.GaussianBlur(inverse_mask_center, (inverse_mask_blur_size, inverse_mask_blur_size), 0)
		temp_frame = inverse_mask_blur_area * inverse_mask_border + (1 - inverse_mask_blur_area) * temp_frame
		temp_frame = temp_frame.clip(0, 255).astype(np.uint8)
		return temp_frame


	def blend_frame(self, temp_frame, paste_frame, blend_percentage):
		face_enhancer_blend = 1 - (blend_percentage / 100)
		temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
		return temp_frame







