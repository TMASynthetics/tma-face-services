import io
import os
import subprocess
import zipfile
from fastapi import Response
import numpy as np
import cv2
import json

from face_services.components.video import Video
from face_services.jobs_database import jobs_database

AUDIO_SIZE_LIMIT_MB = 100
IMAGE_SIZE_LIMIT_MB = 10
VIDEO_SIZE_LIMIT_MB = 1024

AUDIO_MIME_TYPES = ["audio/wav"]
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
VIDEO_MIME_TYPES = ["video/x-msvideo", "video/mp4", "video/mpeg", "video/ogg", "video/webm", "video/3gpp", "video/3gpp2"]


def zipfiles(filenames):
    zip_filename = "archive.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for fpath in filenames:
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)

        # Add file, at correct path
        zf.write(fpath, fname)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp

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

def perform_visual_dubbing(face_visual_dubber):
    jobs_database[face_visual_dubber.id] = {"task": "visual_dubbing", 
                                            'progress': 0, 
                                            'output': None, 
                                            'computing_time_s': 0}
    return face_visual_dubber.run()
    