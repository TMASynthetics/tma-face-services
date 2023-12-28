import os
import sys
sys.path.append(os.getcwd())
from face_services.processors.visual_dubbing.face_visual_dubber import FaceVisualDubber


video_source_path='tests/files/vd1/vd1_source.mp4'
audio_targets_paths=[
    'tests/files/vd1/targets/CO-r21_S_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_ALN_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_CH_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_X_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_TPO_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_KO_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_U_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_G_129_r240P.wav',                                     
    'tests/files/vd1/targets/CO-r21_J_129_r240P.wav',
    'tests/files/vd1/targets/CO-r21_Q_129_r240P.wav',    
    'tests/files/vd1/targets/CO-r21_F_129_r240P.wav',  
    'tests/files/vd1/targets/CO-r21_VL_129_r240P.wav',    
    'tests/files/vd1/targets/CO-r21_AF_129_r240P.wav',
                    ]



# video_source_path='tests/files/vd1/vd1_source.mp4'
# audio_targets_paths=['tests/files/vd1/targets/CO-r21_F_129_r240P.wav']

# video_source_path='tests/files/vd1.mp4'
# audio_targets_paths=['tests/files/vd1.wav','tests/files/vd11.wav','tests/files/vd111.wav']

video_source_path='tests/files/vd1.mp4'
# video_source_path='tests/files/jesus1.png'
audio_targets_paths=['tests/files/vd1.wav']
# audio_targets_paths=['tests/files/vd1/targets/CO-r21_F_129_r240P.wav']

visual_dubber = FaceVisualDubber(video_source_path, audio_targets_paths, model_name="wav2lip")
visual_dubber.run()


print()

