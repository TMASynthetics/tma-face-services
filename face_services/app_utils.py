import os
import subprocess
import numpy as np
import cv2
import json

from face_services.components.video import Video
from face_services.jobs_database import jobs_database

AUDIO_SIZE_LIMIT_MB = 10
IMAGE_SIZE_LIMIT_MB = 10
VIDEO_SIZE_LIMIT_MB = 1024

AUDIO_MIME_TYPES = ["audio/wav"]
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
VIDEO_MIME_TYPES = ["video/x-msvideo", "video/mp4", "video/mpeg", "video/ogg", "video/webm", "video/3gpp", "video/3gpp2"]


# # with open('database.json', 'w') as f:
# #     json.dump({}, f)
# # with open('database.json') as user_file:
# #   jobs_database = json.loads(user_file.read())

# jobs_database = {}


def execute_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr)
        
def encode_frame_to_bytes(frame):
    _, encoded_img = cv2.imencode('.PNG', frame)
    return encoded_img.tobytes()

def decode_frame(content):
    nparr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        ratio = 12
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/ratio, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/ratio, textSize[0][1]
    return 1, 60

def perform_visual_dubbing(face_visual_dubber, visual_dubbing_model):
    jobs_database[face_visual_dubber.id] = {'progress': 0, 'path': None}
    return face_visual_dubber.run(model=visual_dubbing_model)
    




def perform_face_swapping(face_swapper, source_path, target_path, 
                          target_face_ids, source_face_id, face_swapper_model, 
                          face_enhancer_model, enhance, enhancer_blend_percentage):

    jobs_database[face_swapper.id] = {'progress': 0, 'path': None}

    folder_path = os.path.join(os.getcwd(), 'face_services', 'processors', 'temp', face_swapper.id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for folder in ['frames', 'faces', 'audio', 'output', 'debug']:
        if not os.path.exists(os.path.join(folder_path, folder)):
            os.makedirs(os.path.join(folder_path, folder))

    target_media = Video(path=target_path)
    face_swapper.set_source_face(cv2.imread(source_path))

    for frame_idx in range(target_media.frame_number):

        img_target=target_media.get_frame_position_by_index(frame_idx)
        swapped_face = face_swapper.run(img_source=None, 
                                        img_target=img_target, 
                                        target_face_ids=target_face_ids, 
                                        source_face_id=source_face_id,
                                        swapper_model=face_swapper_model,
                                        enhancer_model=face_enhancer_model,
                                        enhance=enhance,
                                        enhancer_blend_percentage=enhancer_blend_percentage)
        
        cv2.imwrite(os.path.join(folder_path, 'output', 'frame_{:0>6}.png'.format(frame_idx)), swapped_face)

        jobs_database[face_swapper.id]['progress'] = np.round(frame_idx/target_media.frame_number, 2)

    if target_media.is_video:
        output_path = target_media.create_video_from_images(os.path.join(folder_path, 'output'), 
                                                            os.path.join('outputs', face_swapper.id + '.mp4'), 
                                                            target_media.fps, target_media.path)
    else:
        output_path = os.path.join('outputs', face_swapper.id + '.png')
        cv2.imwrite(output_path, swapped_face)

    jobs_database[face_swapper.id]['progress'] = 1
    jobs_database[face_swapper.id]['path'] = output_path
    return output_path
