import logging
import os
from typing import List
import cv2
import numpy as np
import subprocess


class Video:
    def __init__(self, path) -> None:
        self.path = path
        self._video = cv2.VideoCapture(self.path)

    @property    
    def is_video(self) -> bool:
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    
    @property    
    def fps(self) -> float:
        return self._video.get(cv2.CAP_PROP_FPS)
    
    @property    
    def frame_number(self) -> int:
        return 1 if int(self._video.get(cv2.CAP_PROP_FRAME_COUNT)) < 0 else int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

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
    
    def get_frames_from_video(self, index_start=1, index_end=None) -> List:
        frames = []
        if not index_end:
            index_end = self.frame_number
        for index in range(index_start, index_end):
            frames.append(self.get_frame_position_by_index(index))
        return frames
    
    def get_frames_from_files(self, folder, index_start=1, index_end=None, file_extension: str='png') -> List:
        frames = []
        if not index_end:
            index_end = self.frame_number
        for index in range(index_start, index_end):
            frames.append(cv2.imread(os.path.join(folder, 'frame_{}.{}'.format(index, file_extension))))
        return frames
    
    @staticmethod  
    def extract_and_save_all_frames(video_path, output_folder, file_extension: str='png'):
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Run FFmpeg command to extract frames
        subprocess.call(['ffmpeg', '-i', video_path, '-q:v', '1', f'{output_folder}/frame_%d.' + Video.check_frame_extension(file_extension)])

    @staticmethod    
    def check_frame_extension(file_extension) -> int:
        return file_extension if file_extension in ['png', 'jpg', 'bmp'] else 'png'


# ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
#   -c:v libx264 -pix_fmt yuv420p out.mp4


    # @staticmethod  
    # def merge_video_and_audio(video_path, audio_path, output_file_path):
    #     subprocess.call(['ffmpeg', '-i', video_path, '-q:v', '1', f'{output_folder}/frame_%d.' + Video.check_frame_extension(file_extension)])
    #     command = ['ffmpeg', "-analyzeduration", "2147483647", 
    #                 "-probesize", "2147483647", 
    #                 "-y", "-i", video_path,
    #                 "-strict", "-2", "-q:v", "1", os.path.join('outputs', face_swapper.id + '.mp4')]
        

