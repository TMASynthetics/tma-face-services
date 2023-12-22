import os
import subprocess
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

def execute_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr)
        
def perform_visual_dubbing(face_visual_dubber, visual_dubbing_model):
    jobs_database[face_visual_dubber.id] = {'progress': 0, 'path': None}
    return face_visual_dubber.run(model=visual_dubbing_model)
    



