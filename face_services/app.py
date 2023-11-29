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
from face_services.app_utils import jobs_database, AUDIO_MIME_TYPES, AUDIO_SIZE_LIMIT_MB, VIDEO_MIME_TYPES, VIDEO_SIZE_LIMIT_MB, decode_frame, encode_frame_to_bytes, get_optimal_font_scale, IMAGE_MIME_TYPES, IMAGE_SIZE_LIMIT_MB, perform_face_swapping, perform_visual_dubbing
from face_services.processors.face_detector import FaceDetector
from face_services.processors.face_anonymizer import FaceAnonymizer
from face_services.processors.face_swapper import FaceSwapper
from face_services.processors.face_enhancer import FaceEnhancer
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
# TESTING
@app.post("/testing/detect", tags=["Testing"])
async def detect(input_file: UploadFile):

    # Get the file size (in bytes)
    input_file.file.seek(0, 2)
    file_size = input_file.file.tell()

    # move the cursor back to the beginning
    await input_file.seek(0)

    # check the content type (MIME type)
    file_content_type = input_file.content_type
    if file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Image too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Get the file
    file_content = await input_file.read()

    frame = decode_frame(file_content)
    detected_faces = app.face_detector.run(frame)

    for detected_face in detected_faces:
        cv2.rectangle(frame,(int(detected_face.bbox[0]), int(detected_face.bbox[1])), (int(detected_face.bbox[2]), int(detected_face.bbox[3])), (0, 255, 0), 2)
        for keypoint in detected_face.kps:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
        for keypoint in detected_face.landmark_2d_106:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (0, 0, 255), -1)
        for keypoint in detected_face.landmark_3d_68:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (0, 255, 255), -1)

        bbox_width = int(detected_face.bbox[2]) - int(detected_face.bbox[0])
        bbox_height = int(detected_face.bbox[3]) - int(detected_face.bbox[1])

        text1 = 'Face ' + str(detected_face.id)
        text2 = 'Sex:' + str(detected_face.sex) + ' Age:' + str(detected_face.age)

        font_scale, font_height = get_optimal_font_scale(text2, bbox_width)
        cv2.putText(frame, text2, (int(detected_face.bbox[0]), int(detected_face.bbox[1]-bbox_height*0.01)), 0, font_scale, (255, 255, 255), 1)
        cv2.putText(frame, text1, (int(detected_face.bbox[0]), int(detected_face.bbox[1]-bbox_height*0.01-font_height)), 0, font_scale, (255, 255, 255), 1)

    return Response(content=encode_frame_to_bytes(frame), media_type="image/png")

@app.post("/testing/anonymize", tags=["Testing"])
async def anonymize(input_file: UploadFile, 
                    face_ids: List[int] = Query(None, description='The ids of the faces to anonymise. Use the detect service to identify the faces.'),
                    method: str | None = Query(default='blur', enum=["blur", "pixelate"], description='The method used to anonymise faces.'), 
                    blur_factor: float = Query(default=3.0, gt=1.0, le=100, description='The blur factor if the anonymisation is perfomed using blurring. Higher values results in less blur.'), 
                    pixel_blocks: int = Query(default=10, ge=1, le=100, description='The number of pixel blocks if the anonymisation is perfomed using pixelisation. Higher values results in finer face.')):

    # Get the file size (in bytes)
    input_file.file.seek(0, 2)
    file_size = input_file.file.tell()

    # move the cursor back to the beginning
    await input_file.seek(0)

    # check the content type (MIME type)
    file_content_type = input_file.content_type
    if file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Image too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Get the file
    file_content = await input_file.read()

    anonymised_face = app.face_anonymiser.run(decode_frame(file_content), face_ids=face_ids, method=method, blur_factor=blur_factor, pixel_blocks=pixel_blocks)
    return Response(content=encode_frame_to_bytes(anonymised_face), media_type="image/png")

@app.post("/testing/swap", tags=["Testing"])
async def swap(source_image_file: UploadFile, target_image_file: UploadFile, 
               source_face_id: int = Query(default=1, ge=1, le=100, description='The id of the face in the source frame use to replace the target face(s). Use the detect service to identify the faces.'),
               target_face_ids: List[int] = Query(None, description='The ids of the faces in the target frame to swap by the source face. Use the detect service to identify the faces.'),
               face_swapper_model: str = Query(default=FaceSwapper().get_available_models()[0], enum=FaceSwapper().get_available_models(), description='The model to use for performing the face swapping.'),    
               enhance: bool = Query(default=False, enum=[False, True], description='Activate in order to enhance the quality of the swapped face(s).'),
               face_enhancer_model: str = Query(default=FaceEnhancer().get_available_models()[0], enum=FaceEnhancer().get_available_models(), description='The model to use for performing the face enhancement.'),
               enhancer_blend_percentage: int = Query(default=100, ge=0, le=100, description='The ratio between the original face and the enhanced one. Higher values results in finer face.')):
                 
    # Get the file size (in bytes)
    target_image_file.file.seek(0, 2)
    file_size = target_image_file.file.tell()
    # move the cursor back to the beginning
    await target_image_file.seek(0)
    # check the content type (MIME type)
    file_content_type = target_image_file.content_type
    if file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Target image too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    else:
        raise HTTPException(status_code=400, detail="Invalid target file type")
    # Get the file
    target_content = await target_image_file.read()


    # Get the file size (in bytes)
    source_image_file.file.seek(0, 2)
    file_size = source_image_file.file.tell()
    # move the cursor back to the beginning
    await source_image_file.seek(0)
    # check the content type (MIME type)
    file_content_type = source_image_file.content_type
    if file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Source image too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    else:
        raise HTTPException(status_code=400, detail="Invalid source file type")
    # Get the file
    source_content = await source_image_file.read()

    swapped_face = app.face_swapper.run(decode_frame(source_content), 
                                    decode_frame(target_content), 
                                    target_face_ids=target_face_ids, 
                                    source_face_id=source_face_id,
                                    swapper_model=face_swapper_model,
                                    enhancer_model=face_enhancer_model,
                                    enhance=enhance,
                                    enhancer_blend_percentage=enhancer_blend_percentage)
    
    return Response(content=encode_frame_to_bytes(swapped_face), media_type="image/png")

@app.post("/testing/enhance", tags=["Testing"])
async def enhance(input_file: UploadFile,
                  face_enhancer_model: str = Query(default=FaceEnhancer().get_available_models()[0], enum=FaceEnhancer().get_available_models(), description='The model to use for performing the face enhancement.'),
                  blend_percentage: int = Query(default=100, ge=0, le=100, description='The ratio between the original face and the enhanced one. Higher values results in finer face.')):

    # Get the file size (in bytes)
    input_file.file.seek(0, 2)
    file_size = input_file.file.tell()

    # move the cursor back to the beginning
    await input_file.seek(0)

    # check the content type (MIME type)
    file_content_type = input_file.content_type
    if file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Image too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Get the file
    file_content = await input_file.read()
    enhanced_face = app.face_enhancer.run(decode_frame(file_content), model=face_enhancer_model, blend_percentage=blend_percentage, )
    return Response(content=encode_frame_to_bytes(enhanced_face), media_type="image/png")
#################################################################



#################################################################
# FACE SWAPPING
@app.post("/swap/process", tags=["Face Swapping"])
async def swap_process(background_tasks: BackgroundTasks, 
               target_video_or_image: UploadFile, 
               source_image_file: UploadFile, 
               source_face_id: int = Query(default=1, ge=1, le=100, description='The id of the face in the source frame use to replace the target face(s). Use the detect service to identify the faces.'),
               target_face_ids: List[int] = Query(None, description='The ids of the faces in the target frame to swap by the source face. Use the detect service to identify the faces.'),
               face_swapper_model: str = Query(default=FaceSwapper().get_available_models()[0], enum=FaceSwapper().get_available_models(), description='The model to use for performing the face swapping.'),    
               enhance: bool = Query(default=False, enum=[False, True], description='Activate in order to enhance the quality of the swapped face(s).'),
               face_enhancer_model: str = Query(default=FaceEnhancer().get_available_models()[0], enum=FaceEnhancer().get_available_models(), description='The model to use for performing the face enhancement.'),
               enhancer_blend_percentage: int = Query(default=100, ge=0, le=100, description='The ratio between the original face and the enhanced one. Higher values results in finer face.')):
                 

    # Get the file size (in bytes)
    target_video_or_image.file.seek(0, 2)
    file_size = target_video_or_image.file.tell()
    # move the cursor back to the beginning
    await target_video_or_image.seek(0)
    # check the content type (MIME type)
    file_content_type = target_video_or_image.content_type

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
    source_image_file.file.seek(0, 2)
    file_size = source_image_file.file.tell()
    # move the cursor back to the beginning
    await source_image_file.seek(0)
    # check the content type (MIME type)
    file_content_type = source_image_file.content_type

    if file_content_type in IMAGE_MIME_TYPES:
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Image file too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB)) 
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    

    # Create temporary video or image file
    target_temp = NamedTemporaryFile(delete=False)
    contents = target_video_or_image.file.read()
    with target_temp as f:
        f.write(contents)
    target_video_or_image.file.close()

    # Create temporary video or image file
    source_temp = NamedTemporaryFile(delete=False)
    contents = source_image_file.file.read()
    with source_temp as f:
        f.write(contents)
    source_image_file.file.close()
    face_swapper = FaceSwapper(swapper_model=face_swapper_model, enhancer_model=face_enhancer_model)
    
    background_tasks.add_task(perform_face_swapping, face_swapper, source_temp.name, target_temp.name, 
                          target_face_ids, source_face_id, face_swapper_model, 
                          face_enhancer_model, enhance, enhancer_blend_percentage)
    return {"id": face_swapper.id}

@app.post("/swap/status", tags=["Face Swapping"])
async def swap_status(id: str): 
    return jobs_database[id]

@app.post("/swap/get", tags=["Face Swapping"])
async def swap_get(id: str): 
    if jobs_database[id]['path']:
        return FileResponse(path=os.path.join('outputs', id + '.mp4'), 
                        filename=id + '.mp4', 
                        media_type='video/mp4',
                        headers=jobs_database[id])
    else:
        return 'Face Swapping processing job {} is not finished. Current progress is {}%.'.format(id, str(int(jobs_database[id]['progress'] * 100)))
#################################################################


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








