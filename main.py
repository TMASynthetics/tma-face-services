import cv2

from processors.face_analyser import FaceAnalyser
from processors.face_enhancer import FaceEnhancer
from processors.face_swapper import FaceSwapper

img_source = cv2.imread('tests/media/sal.png')
img_target = cv2.imread('tests/media/joconde.jpg')

face_analyser = FaceAnalyser()
face_swapper = FaceSwapper()
face_enhancer = FaceEnhancer()


enhanced_face = face_enhancer.run(img_source)
cv2.namedWindow('enhanced_face', 0)
cv2.imshow('enhanced_face', enhanced_face)
cv2.waitKey(1)

swapped_face = face_swapper.run(img_source, img_target)
cv2.namedWindow('swapped_face', 0)
cv2.imshow('swapped_face', swapped_face)
cv2.waitKey(1)

swapped_enhanced_face = face_enhancer.run(swapped_face)
cv2.namedWindow('swapped_enhanced_face', 0)
cv2.imshow('swapped_enhanced_face', swapped_enhanced_face)
cv2.waitKey(1)


print()