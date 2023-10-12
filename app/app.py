from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import os 
from typing import Annotated
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

class Request(BaseModel):
    img_path: str = "images/test.jpg"


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}