import cv2
import mediapipe as mp
import numpy as np
import sqlite3
from datetime import datetime
from database.models import add_face_record, get_all_face_encodings

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class FaceRecognition:
    def __init__(self):
        self.face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    def detect_faces(self, image):
        # image: numpy array (BGR)
        results = self.face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detections = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = (
                    int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                    int(bboxC.width * iw), int(bboxC.height * ih)
                )
                detections.append(bbox)
        return detections
    
    def extract_face_embedding(self, image, bbox):
       
        
        x, y, w, h = bbox
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_embedding = face_img.flatten() / 255.0  
        return face_embedding
    
    def register_face(self, image, name):
        bboxes = self.detect_faces(image)
        if not bboxes:
            return False, "No face detected"
        
        
        embedding = self.extract_face_embedding(image, bboxes[0])
        timestamp = datetime.now().isoformat()
        
        add_face_record(name=name, embedding=embedding.tobytes(), timestamp=timestamp)
        return True, "Face registered"
    
    def recognize_faces(self, image):
        bboxes = self.detect_faces(image)
        known_faces = get_all_face_encodings()  
        
        results = []
        for bbox in bboxes:
            embedding = self.extract_face_embedding(image, bbox)
            name = self.match_face(embedding, known_faces)
            results.append({'bbox': bbox, 'name': name})
        return results
    
    def match_face(self, embedding, known_faces, threshold=0.6):
        # Compare face embedding with known embeddings
        min_dist = float('inf')
        matched_name = "Unknown"
        for record in known_faces:
            known_embedding = np.frombuffer(record['embedding'], dtype=np.float32)
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                matched_name = record['name']
        return matched_name

