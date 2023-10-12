import base64
from fastapi import FastAPI, Response
from pydantic import BaseModel
import cv2
import os 
import numpy as np
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
import uvicorn
from app_utils import decode_frame, encode_frame_to_bytes, serialize_faces_analysis
from processors.face_analyzer import FaceAnalyzer
from processors.face_anonymizer import FaceAnonymizer
from processors.face_enhancer import FaceEnhancer
from processors.face_swapper import FaceSwapper


tags_metadata = [
    {
        "name": "Computer Vision",
        "description": "Computer Vision methods",
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

face_analyzer = FaceAnalyzer()
face_anonymiser = FaceAnonymizer()
face_swapper = FaceSwapper()
face_enhancer = FaceEnhancer()


@app.post("/face/analyze")
async def analyze(image_file: UploadFile):
    image_content = await image_file.read()
    return serialize_faces_analysis(face_analyzer.run(decode_frame(image_content)))


@app.post("/face/anonymize")
async def anonymize(image_file: UploadFile):
    image_content = await image_file.read()
    anonymised_face = face_anonymiser.run(decode_frame(image_content))
    return Response(content=encode_frame_to_bytes(anonymised_face), media_type="image/png")


@app.post("/face/enhance")
async def enhance(image_file: UploadFile):
    image_content = await image_file.read()
    enhanced_face = face_enhancer.run(decode_frame(image_content))
    return Response(content=encode_frame_to_bytes(enhanced_face), media_type="image/png")


@app.post("/face/swap")
async def swap(source_image_file: UploadFile, target_image_file: UploadFile):
    source_content = await source_image_file.read()
    target_centent = await target_image_file.read()
    swapped_face = face_swapper.run(decode_frame(source_content), decode_frame(target_centent))
    return Response(content=encode_frame_to_bytes(swapped_face), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port='8080')