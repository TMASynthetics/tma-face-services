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

class Face:
    def __init__(self, bbox=None, confidence=None) -> None:
        self.bbox = bbox
        self.landmarks_3d_68 = {}
        self.landmarks_2d_106 = {}
        self.confidence = confidence
        self.embedding = None
        self.normed_embedding = None
        self.gender = None
        self.age = None

    @property    
    def fps(self) -> float:
        return self._video.get(cv2.CAP_PROP_FPS)
    
    @property    
    def frame_number(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property    
    def width(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property    
    def height(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
          
    @property    
    def duration(self) -> float:
        return self.frame_number / self.fps

    def get_frame_position_by_time(self, position_ms) -> np.array:
        self._video.set(cv2.CAP_PROP_POS_MSEC, position_ms)
        return self.get_frame()   
    
    def get_frame_position_by_index(self, position_index) -> np.array:
        self._video.set(cv2.CAP_PROP_POS_FRAMES, position_index)
        return self.get_frame()   
    
    def get_current_frame_position(self) -> int:
        return self._video.get(cv2.CAP_PROP_POS_FRAMES)
    
    def get_current_frame_timestamp(self) -> float:
        return self._video.get(cv2.CAP_PROP_POS_MSEC)
    
    def get_frame(self) -> np.array:
        _, frame = self._video.read()
        return frame
    

# class Face(dict):

#     def __init__(self, d=None, **kwargs):
#         if d is None:
#             d = {}
#         if kwargs:
#             d.update(**kwargs)
#         for k, v in d.items():
#             setattr(self, k, v)
#         # Class attributes
#         #for k in self.__class__.__dict__.keys():
#         #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
#         #        setattr(self, k, getattr(self, k))

#     def __setattr__(self, name, value):
#         if isinstance(value, (list, tuple)):
#             value = [self.__class__(x)
#                     if isinstance(x, dict) else x for x in value]
#         elif isinstance(value, dict) and not isinstance(value, self.__class__):
#             value = self.__class__(value)
#         super(Face, self).__setattr__(name, value)
#         super(Face, self).__setitem__(name, value)

#     __setitem__ = __setattr__

#     def __getattr__(self, name):
#         return None

#     @property
#     def embedding_norm(self):
#         if self.embedding is None:
#             return None
#         return l2norm(self.embedding)

#     @property 
#     def normed_embedding(self):
#         if self.embedding is None:
#             return None
#         return self.embedding / self.embedding_norm

#     @property 
#     def sex(self):
#         if self.gender is None:
#             return None
#         return 'M' if self.gender==1 else 'F'
