import logging
import cv2
import os
import sys
sys.path.append(os.getcwd())
from face_services.processors.face_anonymizer import FaceAnonymizer


img_source = cv2.imread('tests/files/sal.jpg')
face_anonymizer = FaceAnonymizer()
anonymized_face = face_anonymizer.run(img_source)

cv2.namedWindow('anonymized_face', 0)
cv2.imshow('anonymized_face', anonymized_face)
cv2.waitKey(1)
print()

