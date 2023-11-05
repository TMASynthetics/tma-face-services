from collections import OrderedDict
import logging
import cv2
import numpy as np

FACIAL_LANDMARKS_3D_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_2D_5_IDXS = OrderedDict([
	("mouth", (2, 3)),
	("right_eye", 0),
	("left_eye", 1),
	("nose", 4),
])

class Face:
    def __init__(self, bbox=None, confidence=None) -> None:
        self.bbox = bbox
        self.landmarks_3d_68 = {}
        self.landmarks_2d_106 = {}
        self.keypoints = {}
        self.confidence = confidence
        self.embedding = None
        self._gender = None
        self.age = None
        self.id = 0
        self.mask = None
        self.segmentation = None
        self.name = None


    @property 
    def sex(self):
        if self._gender is None:
            return None
        return 'M' if self._gender==1 else 'F'
    
    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return np.linalg.norm(self.embedding, ord=2)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm
