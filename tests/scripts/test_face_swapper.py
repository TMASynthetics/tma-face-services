import cv2
import os
import sys
sys.path.append(os.getcwd())
from face_services.processors.face_swapper import FaceSwapper


img_source = cv2.imread('tests/files/sal.jpg')
img_target = cv2.imread('tests/files/monalisa.jpg')
face_swapper = FaceSwapper()
swapped_face = face_swapper.run(img_source, img_target, enhance=True)

cv2.namedWindow('swapped_face', 0)
cv2.imshow('swapped_face', swapped_face)
cv2.waitKey(1)
print()

