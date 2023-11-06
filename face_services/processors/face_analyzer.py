from collections import OrderedDict
import logging
from typing import Any, Dict, Optional, List
import insightface
import uuid
from face_services.models.models_list import FACE_ANALYZER_MODELS
from face_services.typing import Frame


class FaceAnalyzer:
  
	def __init__(self):
		self.id = uuid.uuid4()
		logging.info('FaceAnalyzer {} - Initialize'.format(self.id))
		self.tasks = {}
		self._models = {}
		for task in FACE_ANALYZER_MODELS:
			self.tasks[task] = list(FACE_ANALYZER_MODELS[task].keys())[0]
		self.set_models()



	@staticmethod
	def get_available_tasks():
		return list(FACE_ANALYZER_MODELS.keys())
	
	@staticmethod
	def get_available_models(task):
		return list(FACE_ANALYZER_MODELS[task])
		
	def set_models(self, tasks: Dict=None):
		if tasks:
			self.tasks = tasks
		for task in self.tasks.keys():
			self.tasks[task]
			self._models[task] = 

		return list(FACE_ANALYZER_MODELS.keys())
	
	
	def run(self, frame: Frame):
		logging.info('FaceAnalyzer {} - Run'.format(self.id))

		for task in self.tasks.keys():
			print(task)


		return None

