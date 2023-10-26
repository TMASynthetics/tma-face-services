import logging
import cv2
import numpy as np

class Video:
    def __init__(self, path) -> None:
        self.path = path
        self._video = cv2.VideoCapture(self.path)

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