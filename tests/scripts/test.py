import logging
import cv2
import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from face_services.models.models_list import FACE_ANALYZER_MODELS




logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# from face_services.processors.face_detector import FaceDetector
# from face_services.processors.face_anonymizer import FaceAnonymizer
# from face_services.processors.face_enhancer import FaceEnhancer
# from face_services.processors.face_swapper import FaceSwapper
# from face_services.processors.face_visual_dubber import FaceVisualDubber
# from face_services.processors.face_reenactor import FaceReenactor
from face_services.processors.face_analyzer import FaceAnalyzer

img_source = cv2.imread('tests/files/sal.jpg')
# img_source = cv2.imread('tests/files/noface.jpg')
img_target = cv2.imread('tests/files/monalisa.jpg')
img_group = cv2.imread('tests/files/group.jpg')



face_analyzer = FaceAnalyzer()
face_analyzer.run(img_source, FACE_ANALYZER_MODELS)

# face_analyzer = FaceDetector()
# face_anonymiser = FaceAnonymizer()
# face_swapper = FaceSwapper()
# face_enhancer = FaceEnhancer()
# face_visual_dubber = FaceVisualDubber()
# face_reenactor = FaceReenactor()




# analysed_face = face_analyzer.run(img_source)
# img_source_display = img_source.copy()




# for face in analysed_face:

#     # landmarks = face_analyzer.get_face_3d_features_by_names(face, features_name=['mouth'])
#     # points = []
#     # for keypoint in landmarks:
#     #     cv2.circle(img_source_display, (int(keypoint[0]), int(keypoint[1])), 10, (255, 0, 255), -1)
#     #     points.append([int(keypoint[0]), int(keypoint[1])])

#     # cv2.fillPoly(img_source_display, pts=[np.array(points)], color=(255, 0, 0))

#     cv2.rectangle(img_source_display,(int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), (0, 255, 0), 2)
#     for keypoint in face.kps:
#         cv2.circle(img_source_display, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
#     # for keypoint in face.landmark_2d_106:
#     # for keypoint in face.kps:
#     #     cv2.circle(img_source_display, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
#     # for keypoint in face.landmark_2d_106:
#     #     cv2.circle(img_source_display, (int(keypoint[0]), int(keypoint[1])), 3, (0, 0, 255), -1)
#     # for keypoint in face.landmark_3d_68:
#         # cv2.circle(img_source_display, (int(keypoint[0]), int(keypoint[1])), 3, (0, 255, 255), -1)
#     cv2.putText(img_source_display, 'Sex : ' + str(face.sex), (10, 50), 0, 1, (255, 255, 255), 1)
#     cv2.putText(img_source_display, 'Age : ' + str(face.age), (10, 100), 0, 1, (255, 255, 255), 1)
# # cv2.namedWindow('analysed_face', 0)
# # cv2.imshow('analysed_face', img_source_display)
# # cv2.waitKey(1)


# anonymised_face_blur = face_anonymiser.run(img_group, method="blur", blur_factor=3.0)
# cv2.namedWindow('anonymised_face_blur', 0)
# cv2.imshow('anonymised_face_blur', anonymised_face_blur)
# cv2.waitKey(1)


# anonymised_face_pixelate = face_anonymiser.run(img_group, method="pixelate", pixel_blocks=9)
# cv2.namedWindow('anonymised_face_pixelate', 0)
# cv2.imshow('anonymised_face_pixelate', anonymised_face_pixelate)
# cv2.waitKey(1)


# enhanced_face = face_enhancer.run(img_source, model='codeformer')
# cv2.namedWindow('enhanced_face', 0)
# cv2.imshow('enhanced_face', enhanced_face)
# cv2.waitKey(1)


# swapped_face = face_swapper.run(img_target=img_target, img_source=img_source)
# # swapped_face = face_swapper.run(img_target=img_target, img_source=img_source, swapper_model='inswapper_128_fp16')
# cv2.namedWindow('swapped_face', 0)
# cv2.imshow('swapped_face', swapped_face)
# cv2.waitKey(1)


# swapped_enhanced_face = face_swapper.run(img_source, img_group, enhance=True)
# cv2.namedWindow('swapped_enhanced_face', 0)
# cv2.imshow('swapped_enhanced_face', swapped_enhanced_face)
# cv2.waitKey(1)


# cv2.namedWindow('img_group', 0)
# cv2.imshow('img_group', img_group)
# cv2.waitKey(1)


# cv2.namedWindow('img_source', 0)
# cv2.imshow('img_source', img_source)
# cv2.waitKey(1)


# cv2.namedWindow('img_target', 0)
# cv2.imshow('img_target', img_target)
# cv2.waitKey(1)


print()

