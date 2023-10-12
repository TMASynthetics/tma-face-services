import numpy as np
import cv2

def encode_frame_to_bytes(frame):
    _, encoded_img = cv2.imencode('.PNG', frame)
    return encoded_img.tobytes()

def decode_frame(content):
    nparr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def serialize_faces_analysis(analyzed_faces):
    for analyzed_face in analyzed_faces:
        analyzed_face.bbox = analyzed_face.bbox.tolist()
        analyzed_face.kps = analyzed_face.kps.tolist()
        analyzed_face.embedding = analyzed_face.embedding.tolist()
        analyzed_face.embedding_normed = analyzed_face.normed_embedding.tolist()
        analyzed_face.norm_embedding = float(analyzed_face.embedding_norm)
        analyzed_face.landmark_3d_68 = analyzed_face.landmark_3d_68.tolist()
        analyzed_face.pose = analyzed_face.pose.tolist()
        analyzed_face.landmark_2d_106 = analyzed_face.landmark_2d_106.tolist()
        analyzed_face.det_score = float(analyzed_face.det_score)
        analyzed_face.age = int(analyzed_face.age)
        analyzed_face.gender = int(analyzed_face.gender)
        analyzed_face = analyzed_face.__dict__
    
    return {"analyzed_faces": [analyzed_face.__dict__ for analyzed_face in analyzed_faces]}
