import base64
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import cv2
import os 
import numpy as np
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
import uvicorn
from app_utils import decode_frame, encode_frame_to_bytes, serialize_faces_analysis
from processors.face_detector import FaceDetector
from processors.face_anonymizer import FaceAnonymizer
# from processors.face_enhancer import FaceEnhancer
# from processors.face_swapper import FaceSwapper


tags_metadata = [
    {
        "name": "Face Services",
        "description": "Face Services",
        # "externalDocs": {
        #     "description": "Items external docs",
        #     "url": "https://fastapi.tiangolo.com/",
        # },
    },
]
app = FastAPI(
    title="TMA - Synthetic Media Team - Face Services",
    description="",
    version="0.0.1",
    # terms_of_service="http://example.com/terms/",
    contact={
        "name": "Thierry SAMMOUR",
        # "url": "http://x-force.example.com/contact/",
        "email": "tsammour@bethel.jw.org",
    },
    openapi_tags=tags_metadata
)

face_analyzer = FaceDetector()
face_anonymiser = FaceAnonymizer()
# face_swapper = FaceSwapper()
# face_enhancer = FaceEnhancer()


IMAGE_SIZE_LIMIT_MB = 1
VIDEO_SIZE_LIMIT_MB = 1024
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
VIDEO_MIME_TYPES = ["video/x-msvideo", "video/mpeg", "video/ogg", "video/webm", "video/3gpp", "video/3gpp2"]


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/face/detect", tags=["Face Services"])
async def detect(input_file: UploadFile):

    # Get the file size (in bytes)
    input_file.file.seek(0, 2)
    file_size = input_file.file.tell()

    # move the cursor back to the beginning
    await input_file.seek(0)

    # check the content type (MIME type)
    file_type = None
    file_content_type = input_file.content_type
    if file_content_type in IMAGE_MIME_TYPES:
        file_type = "image"
        if file_size > IMAGE_SIZE_LIMIT_MB * 1024 * 1024:
            # more than IMAGE_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Image too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    elif file_content_type in VIDEO_MIME_TYPES:
        file_type = "video"
        if file_size > VIDEO_SIZE_LIMIT_MB * 1024 * 1024:
            # more than VIDEO_SIZE_LIMIT_MB MB
            raise HTTPException(status_code=400, detail="Video too large, maximum image size is {}MB".format(IMAGE_SIZE_LIMIT_MB))
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Get the file
    file_content = await input_file.read()
    response = {"filename": input_file.filename, 
                "media_type": file_type,
                "detected_faces": serialize_faces_analysis(face_analyzer.run(decode_frame(file_content)))}
    return response


@app.post("/face/anonymize")
async def anonymize(image_file: UploadFile):
    image_content = await image_file.read()
    anonymised_face = face_anonymiser.run(decode_frame(image_content), face_ids=[1])
    return Response(content=encode_frame_to_bytes(anonymised_face), media_type="image/png")


# @app.post("/face/enhance")
# async def enhance(image_file: UploadFile):
#     image_content = await image_file.read()
#     enhanced_face = face_enhancer.run(decode_frame(image_content))
#     return Response(content=encode_frame_to_bytes(enhanced_face), media_type="image/png")


# @app.post("/face/swap")
# async def swap(source_image_file: UploadFile, target_image_file: UploadFile):
#     source_content = await source_image_file.read()
#     target_centent = await target_image_file.read()
#     swapped_face = face_swapper.run(decode_frame(source_content), decode_frame(target_centent))
#     return Response(content=encode_frame_to_bytes(swapped_face), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port='8080')