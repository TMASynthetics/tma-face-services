from collections import OrderedDict
from face_services.logger import logger
from typing import Any, Optional, List, Tuple
import uuid
from face_services.components.face import Face
from face_services.models_list import FACE_ANALYZER_MODELS
import numpy as np
import cv2
from face_services.processors.face_helper import resize_frame_dimension, warp_face
from face_services.typing import Embedding, Frame, Kps
import onnxruntime 
from face_services.processors.utilities import onnx_providers


class FaceDetector:
  
	def __init__(self):
		self.id = uuid.uuid4()
		logger.info('FaceDetector {} - Initialize'.format(self.id))
		self.model = cv2.FaceDetectorYN.create(FACE_ANALYZER_MODELS['detection']['face_detection_yunet']['path'], None, (0, 0))
		self.face_recognizer = onnxruntime.InferenceSession(FACE_ANALYZER_MODELS['recognition']['face_recognition_arcface_inswapper']['path'], 
													  providers=onnx_providers)

	def run(self, frame):
		logger.info('FaceDetector {} - Run'.format(self.id))

		faces: List[Face] = []

		# preprocessing
		temp_frame = resize_frame_dimension(frame, 1024, 1024)
		temp_frame_height, temp_frame_width, _ = temp_frame.shape
		frame_height, frame_width, _ = frame.shape
		ratio_height = frame_height / temp_frame_height
		ratio_width = frame_width / temp_frame_width
		self.model.setScoreThreshold(0.5)
		self.model.setNMSThreshold(0.5)
		self.model.setTopK(100)
		self.model.setInputSize((temp_frame_width, temp_frame_height))

		# inference
		_, detections = self.model.detect(temp_frame)

		# postprocessing
		if detections.any():
			for detection in detections:
				bbox =\
				[
					detection[0:4][0] * ratio_width,
					detection[0:4][1] * ratio_height,
					(detection[0:4][0] + detection[0:4][2]) * ratio_width,
					(detection[0:4][1] + detection[0:4][3]) * ratio_height
				]
				face = Face(bbox=bbox, confidence=detection[14])
				face.keypoints = (detection[4:14].reshape((5, 2)) * [[ ratio_width, ratio_height ]]).tolist()
				# face.embedding = self.calc_embedding(temp_frame, face.keypoints)
				faces.append(face)
				
		return self.identify_faces(faces)


	def calc_embedding(self, temp_frame : Frame, kps : Kps) -> Tuple[Embedding, Embedding]:
		crop_frame, matrix = warp_face(temp_frame, kps, 'arcface_v2', (112, 112))
		crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
		crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
		crop_frame = np.expand_dims(crop_frame, axis = 0)
		embedding = self.face_recognizer.run(None,
		{
			self.face_recognizer.get_inputs()[0].name: crop_frame
		})[0]
		embedding = embedding.ravel()
		return embedding

	def identify_faces(self, detected_faces):
		# logger.info('FaceDetector {} - Identify faces'.format(self.id))
		for idx, detected_face in enumerate(detected_faces):
			detected_face.id = idx + 1
		return detected_faces

	@staticmethod
	def get_face_by_id(detected_faces, id):
		for detected_face in detected_faces:
			if detected_face.id == id:
				return detected_face
		return None


