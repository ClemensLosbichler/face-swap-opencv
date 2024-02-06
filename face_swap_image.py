import cv2
import dlib
import numpy as np

from face_swap_helper import FaceSwapHelper

class FaceSwapImage:
    def __init__(self, image):
        self.image = image
        self.first_face = None
        self.landmark_points = None
        self.landmark_point_indices = None
        self.head_tilt = None

    def copy(self):
        face_swap_image = FaceSwapImage(self.image)
        if self.first_face is not None:
            face_swap_image.first_face = self.first_face
        if self.landmark_points is not None:
            face_swap_image.landmark_points = self.landmark_points.copy()
        if self.landmark_point_indices is not None:
            face_swap_image.landmark_point_indices = self.landmark_point_indices.copy()
        if self.head_tilt is not None:
            face_swap_image.head_tilt = self.head_tilt
        return face_swap_image
    
    def detect_first_face(self, frontal_face_detector):
        self.first_face = FaceSwapHelper.detect_first_face(self.image, frontal_face_detector)
        if self.first_face is None:
            raise RuntimeError("no face detected")
    
    def detect_landmark_points(self, facial_landmark_predictor):
        if self.first_face is None:
            return None
        self.landmark_points = FaceSwapHelper.detect_facial_landmarks(
            self.image, self.first_face, facial_landmark_predictor)
        if len(self.landmark_points) == 0: 
            raise RuntimeError("no landmarks detected")

    def calculate_landmark_point_indices(self):
        if self.landmark_points is None:
            return None
        triangles = FaceSwapHelper.triangulate_landmark_points(self.landmark_points)
        triangle_indices = FaceSwapHelper.index_landmark_point_triangles(
            self.landmark_points, triangles)
        self.landmark_point_indices = triangle_indices

    def calculate_head_tilt(self):
        if self.landmark_points is None:
            return None
        self.head_tilt = FaceSwapHelper.get_head_tilt_from_landmarks(self.landmark_points)