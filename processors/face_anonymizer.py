from typing import Any, Optional, List
import cv2
import numpy as np
from processors.face_detector import FaceDetector


class FaceAnonymizer:
  
	def __init__(self):
		self.face_detector = FaceDetector()

	def run(self, frame, face_ids=[], method="blur", blur_factor=3.0, pixel_blocks=9):

		anonymised_frame = frame.copy()
		detected_faces = self.face_detector.run(anonymised_frame)

		if method not in ["blur", "pixelate"]:
			method = "blur"
		if face_ids is None:
			face_ids=[]

		for face in detected_faces:
			if face.id in face_ids or len(face_ids)==0 :
				cropped_face = anonymised_frame[int(face.bbox[1]):int(face.bbox[3]), int(face.bbox[0]):int(face.bbox[2])]
				if method == "blur":
					anonymised_face = self.anonymize_face_blured(cropped_face, blur_factor)
				else:
					anonymised_face = self.anonymize_face_pixelate(cropped_face, pixel_blocks)
				anonymised_frame[int(face.bbox[1]):int(face.bbox[3]), int(face.bbox[0]):int(face.bbox[2])] = anonymised_face
		return anonymised_frame


	def anonymize_face_blured(self, cropped_face, blur_factor):
		(h, w) = cropped_face.shape[:2]
		kW = int(w / blur_factor)
		kH = int(h / blur_factor)
		if kW % 2 == 0:
			kW -= 1
		if kH % 2 == 0:
			kH -= 1
		return cv2.GaussianBlur(cropped_face, (kW, kH), 0)


	def anonymize_face_pixelate(self, cropped_face, pixel_blocks):
		(h, w) = cropped_face.shape[:2]
		xSteps = np.linspace(0, w, pixel_blocks + 1, dtype="int")
		ySteps = np.linspace(0, h, pixel_blocks + 1, dtype="int")
		for i in range(1, len(ySteps)):
			for j in range(1, len(xSteps)):
				startX = xSteps[j - 1]
				startY = ySteps[i - 1]
				endX = xSteps[j]
				endY = ySteps[i]
				roi = cropped_face[startY:endY, startX:endX]
				(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
				cv2.rectangle(cropped_face, (startX, startY), (endX, endY), (B, G, R), -1)
		return cropped_face








