import glob
import logging
import os
import sys

sys.path.append(os.getcwd())

from face_services.components.video import Video
from face_services.components.audio import Audio



# Video.extract_and_save_sample('tests/files/vd1/CO-r21_E_129_r720P.mp4', 'tests/files/vd1/vd1_source.mp4', 1300, 61300)

audio = Audio('tests/files/ALN.wav')

# for path in glob.glob('tests/files/vd1/originals/*.mp4'):   
#     Video.extract_and_save_sample(path, path.replace(path.split('/')[-2], 'targets'), 1300, 61300)
#     audio_path = Video.extract_audio_from_video(video_path=path.replace(path.split('/')[-2], 'targets'), extracted_audio_folder='tests/files/vd1/targets')


# video = Video('tests/files/vd1.mp4')

# video.extract_and_save_all_frames(video_path=video.path, output_folder='tests/files/vd1')

# # audio_path = video.extract_audio_from_video(video_path=video.path, extracted_audio_folder='tests/files/vd1')

# # video.add_audio_to_video(video_path=video.path, audio_path=audio_path, output_video_path='tests/files/vd1/vd1.mp4')

# video.create_video_from_images('tests/files/vd1', 
#                                output_video_path='tests/files/vd1/vd1_reconstructed.mp4', 
#                                fps=video.fps, audio_path='tests/files/vd_fr.wav')

# # frames = video.get_frames_from_video(1, 10)

# # frames = video.get_frames_from_files(folder='tests/files/vd1', index_start=1, index_end=None, file_extension='png')

print()
