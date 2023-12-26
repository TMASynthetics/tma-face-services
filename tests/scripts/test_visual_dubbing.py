import logging
import os
import sys
sys.path.append(os.getcwd())

# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from face_services.processors.face_visual_dubber import FaceVisualDubber

# face_visual_dubber = FaceVisualDubber(video_source_path='tests/files/vd1.mp4', 
#                   audio_target_path='tests/files/vd1.wav')
# face_visual_dubber.run()

face_visual_dubber = FaceVisualDubber(video_source_path='tests/files/monalisa.jpg', 
                  audio_target_path='tests/files/vd1.wav')
face_visual_dubber.run()

print()

