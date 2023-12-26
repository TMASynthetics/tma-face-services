import logging
import cv2
import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from face_services.processors.face_detector import FaceDetector


img_source = cv2.imread('tests/files/group.jpg')
face_detector = FaceDetector()


analysed_face = face_detector.run(img_source)
img_source_display = img_source.copy()

for face in analysed_face:
    points = []
    for keypoint in face.keypoints:
        cv2.circle(img_source_display, (int(keypoint[0]), int(keypoint[1])), 2, (255, 0, 255), -1)
        points.append([int(keypoint[0]), int(keypoint[1])])
    cv2.rectangle(img_source_display,(int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), (0, 255, 0), 2)

  
cv2.namedWindow('analysed_face', 0)
cv2.imshow('analysed_face', img_source_display)
cv2.waitKey(1)
print()

