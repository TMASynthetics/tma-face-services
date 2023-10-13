import numpy as np
import cv2

def encode_frame_to_bytes(frame):
    _, encoded_img = cv2.imencode('.PNG', frame)
    return encoded_img.tobytes()

def decode_frame(content):
    nparr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def serialize_faces_analysis(detected_faces):
    for detected_face in detected_faces:
        detected_face.bbox = detected_face.bbox.tolist()
        detected_face.kps = detected_face.kps.tolist()
        detected_face.embedding = detected_face.embedding.tolist()
        detected_face.embedding_normed = detected_face.normed_embedding.tolist()
        detected_face.norm_embedding = float(detected_face.embedding_norm)
        detected_face.landmark_3d_68 = detected_face.landmark_3d_68.tolist()
        detected_face.pose = detected_face.pose.tolist()
        detected_face.landmark_2d_106 = detected_face.landmark_2d_106.tolist()
        detected_face.det_score = float(detected_face.det_score)
        detected_face.age = int(detected_face.age)
        detected_face.gender = int(detected_face.gender)
        detected_face = detected_face.__dict__
    
    return [detected_face.__dict__ for detected_face in detected_faces]
