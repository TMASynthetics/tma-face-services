from typing import Any, Dict, Optional, List, Tuple
import numpy
from face_services.processors.utilities import resolve_relative_path
from ..processors.face_detector import FaceDetector
import logging
import onnxruntime
import cv2

FACE_ENHANCER_MODELS =\
{
	'codeformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/codeformer.onnx')
	},
	'gfpgan_1.2':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.2.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GFPGANv1.2.onnx')
	},
	'gfpgan_1.3':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.3.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GFPGANv1.3.onnx')
	},
	'gfpgan_1.4':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.4.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GFPGANv1.4.onnx')
	},
	'gpen_bfr_512':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GPEN-BFR-512.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GPEN-BFR-512.onnx')
	}
}

class FaceEnhancer:
  
	def __init__(self, model='gfpgan_1.4'):
		logging.info('FaceEnhancer - Initialize')
		self.model = onnxruntime.InferenceSession(FACE_ENHANCER_MODELS[model]['path'], providers = ['CPUExecutionProvider'])
		self.face_analyzer = FaceDetector()

	def run(self, frame, blend_percentage: int=100):
		logging.info('FaceEnhancer - Run')
		enhanced_frame = frame.copy()
		analysed_faces = self.face_analyzer.run(frame)

		for face in analysed_faces:
			crop_frame, affine_matrix = self.warp_face(face, enhanced_frame)
			crop_frame = self.prepare_crop_frame(crop_frame)
			frame_processor_inputs = {}
			frame_processor_inputs['input'] = crop_frame
			crop_frame = self.model.run(None, frame_processor_inputs)[0][0]
			crop_frame = self.normalize_crop_frame(crop_frame)
			paste_frame = self.paste_back(enhanced_frame, crop_frame, affine_matrix)
			enhanced_frame = self.blend_frame(enhanced_frame, paste_frame, blend_percentage)

		return enhanced_frame



	def warp_face(self, target_face, temp_frame):
		template = numpy.array(
		[
			[ 192.98138, 239.94708 ],
			[ 318.90277, 240.1936 ],
			[ 256.63416, 314.01935 ],
			[ 201.26117, 371.41043 ],
			[ 313.08905, 371.15118 ]
		])
		affine_matrix = cv2.estimateAffinePartial2D(target_face['kps'], template, method = cv2.LMEDS)[0]
		crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (512, 512))
		return crop_frame, affine_matrix


	def prepare_crop_frame(self, crop_frame):
		crop_frame = crop_frame[:, :, ::-1] / 255.0
		crop_frame = (crop_frame - 0.5) / 0.5
		crop_frame = numpy.expand_dims(crop_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
		return crop_frame


	def normalize_crop_frame(self, crop_frame):
		crop_frame = numpy.clip(crop_frame, -1, 1)
		crop_frame = (crop_frame + 1) / 2
		crop_frame = crop_frame.transpose(1, 2, 0)
		crop_frame = (crop_frame * 255.0).round()
		crop_frame = crop_frame.astype(numpy.uint8)[:, :, ::-1]
		return crop_frame


	def paste_back(self, temp_frame, crop_frame, affine_matrix):
		inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
		temp_frame_height, temp_frame_width = temp_frame.shape[0:2]
		crop_frame_height, crop_frame_width = crop_frame.shape[0:2]
		inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
		inverse_mask = numpy.ones((crop_frame_height, crop_frame_width, 3), dtype = numpy.float32)
		inverse_mask_frame = cv2.warpAffine(inverse_mask, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
		inverse_mask_frame = cv2.erode(inverse_mask_frame, numpy.ones((2, 2)))
		inverse_mask_border = inverse_mask_frame * inverse_crop_frame
		inverse_mask_area = numpy.sum(inverse_mask_frame) // 3
		inverse_mask_edge = int(inverse_mask_area ** 0.5) // 20
		inverse_mask_radius = inverse_mask_edge * 2
		inverse_mask_center = cv2.erode(inverse_mask_frame, numpy.ones((inverse_mask_radius, inverse_mask_radius)))
		inverse_mask_blur_size = inverse_mask_edge * 2 + 1
		inverse_mask_blur_area = cv2.GaussianBlur(inverse_mask_center, (inverse_mask_blur_size, inverse_mask_blur_size), 0)
		temp_frame = inverse_mask_blur_area * inverse_mask_border + (1 - inverse_mask_blur_area) * temp_frame
		temp_frame = temp_frame.clip(0, 255).astype(numpy.uint8)
		return temp_frame


	def blend_frame(self, temp_frame, paste_frame, blend_percentage):
		face_enhancer_blend = 1 - (blend_percentage / 100)
		temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
		return temp_frame







