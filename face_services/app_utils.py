import numpy as np
import cv2

AUDIO_SIZE_LIMIT_MB = 10
IMAGE_SIZE_LIMIT_MB = 10
VIDEO_SIZE_LIMIT_MB = 1024

AUDIO_MIME_TYPES = ["audio/mp4", "audio/wav", "audio/x-wav", "audio/mpeg"]
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
VIDEO_MIME_TYPES = ["video/x-msvideo", "video/mp4", "video/mpeg", "video/ogg", "video/webm", "video/3gpp", "video/3gpp2"]

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
    dubbed_video_path = face_visual_dubber.run(model=visual_dubbing_model)