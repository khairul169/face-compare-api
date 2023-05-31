import cv2
import numpy as np
import dlib

class FaceRecognition:
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    
    def get_face_descriptor(self, img_path):
        # Read the Input Image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return False

        landmarks = self.predictor(gray, faces[0])
        return np.array(self.face_encoder.compute_face_descriptor(img, landmarks, num_jitters=1));

    def compare_face(self, main, face_to_compare):
        return 1.0-np.linalg.norm(main - face_to_compare);


