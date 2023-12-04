import cv2
import os
import sys
sys.path.append(os.getcwd())
from face_services.processors.face_enhancer import FaceEnhancer

from face_services.logger import logger


img_source = cv2.imread('tests/files/sal.jpg')
face_enhancer = FaceEnhancer()


enhanced_face = face_enhancer.run(img_source, model='codeformer')

cv2.namedWindow('enhanced_face', 0)
cv2.imshow('enhanced_face', enhanced_face)
cv2.waitKey(1)
print()

