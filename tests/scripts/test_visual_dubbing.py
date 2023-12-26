import logging
import os
import sys
sys.path.append(os.getcwd())
import time

# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from face_services.processors.face_visual_dubber import FaceVisualDubber

time_start = time.time()

face_visual_dubber = FaceVisualDubber(video_source_path='tests/files/vd1/vd1_source.mp4', 
                  audio_target_path='tests/files/vd1/targets/CO-r21_F_129_r240P.wav')
face_visual_dubber.run()

# face_visual_dubber = FaceVisualDubber(video_source_path='tests/files/vd1.mp4', 
#                   audio_target_path='tests/files/vd1.wav')
# face_visual_dubber.run()

print(time.time() - time_start)


print()

