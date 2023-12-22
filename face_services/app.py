import base64
import json
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import os 
import numpy as np
from typing import Annotated, List
from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile

from face_services.logger import logger
from face_services.app_utils import jobs_database, AUDIO_MIME_TYPES, AUDIO_SIZE_LIMIT_MB, VIDEO_MIME_TYPES, VIDEO_SIZE_LIMIT_MB, IMAGE_MIME_TYPES, IMAGE_SIZE_LIMIT_MB, perform_visual_dubbing
from face_services.processors.face_visual_dubber import FaceVisualDubber

tags_metadata = [
    {
        "name": "Face Services API",
        "description": "This API provides human face processing capabilities.",
        "externalDocs": {
            "description": "Documentation",
            "url": "https://jwsite.sharepoint.com/:f:/r/sites/WHQ-MEPS-TMASyntheticMedia-Team/Shared%20Documents/Products/Face%20Services%20API?csf=1&web=1&e=IVOU8p",
        },
    },
]

app = FastAPI(
    title="TMA - Synthetic Media Team - Face Services API",
    description="",
    version="1.0",
    contact={
        "name": "Thierry SAMMOUR",
        "email": "tsammour@bethel.jw.org",
    },
    openapi_tags=tags_metadata
)

@app.route('/', include_in_schema=False)
def app_redirect(_):
    return RedirectResponse(url='/docs')

@app.on_event("startup")
async def startup_event():
    logger.info('Start')


#################################################################
# VISUAL DUBBING
@app.post("/visual_dubbing/process", tags=["Visual Dubbing"])
async def visual_dubbing_process(input_video_or_image: UploadFile, 
                         target_audio: UploadFile,
                         background_tasks: BackgroundTasks,
                         visual_dubbing_model: str = Query(default=FaceVisualDubber().get_available_models()[0], 
                                                           enum=FaceVisualDubber().get_available_models(), description='The model to use for performing the visual dubbing.')):

    # Get the file size (in bytes)
    input_video_or_image.file.seek(0, 2)
    file_size = input_video_or_image.file.tell()
    # move the cursor back to the beginning
    await input_video_or_image.seek(0)
    # check the content type (MIME type)
    file_content_type = input_video_or_image.content_type

    if file_content_type in VIDEO_MIME_TYPES:
        if file_size > VIDEO_SIZE_LIMIT_MB * 1024 * 1024:
            # more than VIDEO_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Video file too large, maximum video size is {}MB".format(VIDEO_SIZE_LIMIT_MB))
    elif file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Image file too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB)) 
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")


    # Get the file size (in bytes)
    target_audio.file.seek(0, 2)
    file_size = target_audio.file.tell()
    # move the cursor back to the beginning
    await target_audio.seek(0)
    # check the content type (MIME type)
    file_content_type = target_audio.content_type

    if file_content_type in AUDIO_MIME_TYPES:
        if file_size > AUDIO_SIZE_LIMIT_MB * 1024 * 1024:
            # more than AUDIO_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Audio file too large, maximum audio size is {}MB".format(AUDIO_SIZE_LIMIT_MB)) 
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Create temporary video or image file
    video_temp = NamedTemporaryFile(delete=False)
    contents = input_video_or_image.file.read()
    with video_temp as f:
        f.write(contents)
    input_video_or_image.file.close()

    # Create temporary audio file
    audio_temp = NamedTemporaryFile(delete=False)
    contents = target_audio.file.read()
    with audio_temp as f:
        f.write(contents)
    target_audio.file.close()

    face_visual_dubber = FaceVisualDubber(video_source_path=video_temp.name, 
                                          audio_target_path=audio_temp.name)
    
    background_tasks.add_task(perform_visual_dubbing, face_visual_dubber, visual_dubbing_model)
    return {"id": face_visual_dubber.id}

@app.post("/visual_dubbing/status", tags=["Visual Dubbing"])
async def visual_dubbing_status(id: str): 
    return jobs_database[id] if id in jobs_database.keys() else 'No id : {}'.format(id)

@app.post("/visual_dubbing/get", tags=["Visual Dubbing"])
async def visual_dubbing_get(id: str): 
    if jobs_database[id]['path']:
        return FileResponse(path=os.path.join('outputs', id + '.mp4'), 
                        filename=id + '.mp4', 
                        media_type='video/mp4')
    else:
        return 'Visual Dubbing processing job {} is not finished. Current progress is {}%.'.format(id, str(int(jobs_database[id]['progress'] * 100)))
#################################################################








