import base64
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
import cv2
import os 
import numpy as np
from typing import Annotated, List
from fastapi import FastAPI, File, UploadFile
import uvicorn
from face_services.app_utils import decode_frame, encode_frame_to_bytes, get_optimal_font_scale, serialize_faces_analysis
from face_services.processors.face_detector import FaceDetector
from face_services.processors.face_anonymizer import FaceAnonymizer
from face_services.processors.face_enhancer import FaceEnhancer
from face_services.processors.face_swapper import FaceSwapper


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
    version="0.0.1",
    contact={
        "name": "Thierry SAMMOUR",
        "email": "tsammour@bethel.jw.org",
    },
    openapi_tags=tags_metadata
)

face_analyzer = FaceDetector()
face_anonymiser = FaceAnonymizer()
face_swapper = FaceSwapper()
face_enhancer = FaceEnhancer()


IMAGE_SIZE_LIMIT_MB = 10
VIDEO_SIZE_LIMIT_MB = 1024
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
VIDEO_MIME_TYPES = ["video/x-msvideo", "video/mp4", "video/mpeg", "video/ogg", "video/webm", "video/3gpp", "video/3gpp2"]


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
    detected_faces = face_analyzer.run(frame)

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

    anonymised_face = face_anonymiser.run(decode_frame(file_content), face_ids=face_ids, method=method, blur_factor=blur_factor, pixel_blocks=pixel_blocks)
    return Response(content=encode_frame_to_bytes(anonymised_face), media_type="image/png")


@app.post("/testing/swap", tags=["Testing"])
async def swap(source_image_file: UploadFile, target_image_file: UploadFile, 
               source_face_id: int = Query(default=1, ge=1, le=100, description='The id of the face in the source frame use to replace the target face(s). Use the detect service to identify the faces.'),
               target_face_ids: List[int] = Query(None, description='The ids of the faces in the target frame to swap by the source face. Use the detect service to identify the faces.'),
               enhance: bool = Query(default=False, enum=[False, True], description='Activate in order to enhance the quality of the swapped face(s).')):

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

    swapped_face = face_swapper.run(decode_frame(source_content), 
                                    decode_frame(target_content), 
                                    enhance=enhance, 
                                    target_face_ids=target_face_ids, 
                                    source_face_id=source_face_id)
    
    return Response(content=encode_frame_to_bytes(swapped_face), media_type="image/png")


@app.post("/testing/enhance", tags=["Testing"])
async def enhance(input_file: UploadFile):

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
    enhanced_face = face_enhancer.run(decode_frame(file_content))
    return Response(content=encode_frame_to_bytes(enhanced_face), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)