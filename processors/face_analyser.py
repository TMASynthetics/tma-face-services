from typing import Any, Optional, List
import insightface


class FaceAnalyser:
  
	def __init__(self):
		self.model = insightface.app.FaceAnalysis(name = 'buffalo_l', root='.assets')
		self.model.prepare(ctx_id = 0)

	def run(self, frame):
		return self.model.get(frame)












