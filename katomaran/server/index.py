from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from io import BytesIO
from ai.face_recognition import FaceRecognition

app = FastAPI()
face_recog = FaceRecognition()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_imagefile(file) -> np.ndarray:
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

@app.post("/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    img = read_imagefile(await file.read())
    success, message = face_recog.register_face(img, name)
    return {"success": success, "message": message}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    img = read_imagefile(await file.read())
    results = face_recog.recognize_faces(img)
    
    return {"results": results}
