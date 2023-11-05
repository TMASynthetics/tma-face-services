from collections import OrderedDict
import logging
from typing import Any, Optional, List
import insightface
import uuid
from face_services.models.models_list import FACE_ANALYZER_MODELS


class FaceAnalyzer:
  
	def __init__(self):
		logging.info('FaceAnalyzer - Initialize')
		self.model = None
		self.tasks = {}
		self.id = uuid.uuid4()


	@staticmethod
	def get_available_tasks():
		return list(FACE_ANALYZER_MODELS.keys())
	
	@staticmethod
	def get_available_models(task):
		return list(FACE_ANALYZER_MODELS[task])
		
	def run(self, frame, tasks=None):
		logging.info('FaceAnalyzer - Run')

		for task in self.tasks.keys():
			print(task)


		return None

