from typing import Any, Optional, List
import threading
import insightface
import numpy
from gfpgan.utils import GFPGANer
from processors.face_detector import FaceDetector
import os
import gfpgan

class FaceEnhancer:
  
	def __init__(self):
		self.model = gfpgan.GFPGANer(model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../.assets/models/GFPGANv1.4.pth')), upscale = 1)
		self.face_analyzer = FaceDetector()

	def run(self, frame):

		enhanced_frame = frame.copy()
		analysed_faces = self.face_analyzer.run(frame)

		for face in analysed_faces:
			start_x, start_y, end_x, end_y = map(int, face['bbox'])
			padding_x = int((end_x - start_x) * 0.5)
			padding_y = int((end_y - start_y) * 0.5)
			start_x = max(0, start_x - padding_x)
			start_y = max(0, start_y - padding_y)
			end_x = max(0, end_x + padding_x)
			end_y = max(0, end_y + padding_y)
			crop_frame = enhanced_frame[start_y:end_y, start_x:end_x]

			_, _, crop_frame = self.model.enhance(crop_frame, paste_back = True)
			enhanced_frame[start_y:end_y, start_x:end_x] = crop_frame
	
		return enhanced_frame








