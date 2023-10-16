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

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        ratio = 12
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/ratio, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/ratio, textSize[0][1]
    return 1, 60
