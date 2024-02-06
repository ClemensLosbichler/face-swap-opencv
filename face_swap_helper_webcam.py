import cv2
import dlib
import numpy as np

from face_swap_image import FaceSwapImage
from face_swap_helper import FaceSwapHelper

MODEL_PATH = "data/shape_predictor_68_face_landmarks.dat"
CASCADE_CLASSIFIER_PATH = "data/haarcascade_frontalface_default.xml"

class FaceSwapHelperWebcam:
    def __init__(self):
        self.set_webcam(0)

        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.facial_landmark_predictor = dlib.shape_predictor(MODEL_PATH)
        self.face_cascade_classifier = cv2.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

        self.webcam_image = FaceSwapImage(None)
        self.webcam_image_prev = FaceSwapImage(None)

        self.webcam_head_tilt = 1
        self.webcam_landmark_points = 0
        self.webcam_landmark_points_prev = 0
        self.landmarks_history = []

        self.paused = True
        self.successful = False
        self.flip_camera = False

    def __del__(self):
        self.release_webcam()

    def set_webcam(self, webcam_number):
        self.release_webcam()
        self.webcam = cv2.VideoCapture(webcam_number)

        if not self.webcam.isOpened():
            print("Cannot open WebCam")
            quit()

    def release_webcam(self):
        try:
            self.webcam.release()
        except:
            pass

    def set_image(self, image):
        image = FaceSwapImage(image)
        image.detect_first_face(self.frontal_face_detector)
        image.detect_landmark_points(self.facial_landmark_predictor)
        image.calculate_landmark_point_indices()
        return image

    def set_source_image(self, path):
        source_image_cv = cv2.imread(path)
        if source_image_cv is None:
            print("Image path is wrong")
            quit()
        self.source_image = self.set_image(source_image_cv)
        self.source_image_tilted = self.set_image(cv2.flip(self.source_image.image, 1))

    def stop_abrupt_landmark_movements(self):
        total_distance = 0

        for i in range(len(self.webcam_image.landmark_points)):
            total_distance += abs(self.webcam_image.landmark_points[i][0] - 
                self.webcam_image_prev.landmark_points[i][0])
            total_distance += abs(self.webcam_image.landmark_points[i][1] - 
                self.webcam_image_prev.landmark_points[i][1])

        max_distance = abs((self.webcam_image.landmark_points[1][0] - 
            self.webcam_image.landmark_points[17][0])) * 4
        if total_distance / 68 > max_distance:
            self.webcam_image.landmark_points = (
                self.webcam_image_prev.landmark_points.copy())

    def smooth_landmark_movement(self):
        if len(self.landmarks_history) == 2:
            for i in range(len(self.landmarks_history) - 1):
                self.landmarks_history[i] = self.landmarks_history[i+1]
            self.landmarks_history[len(self.landmarks_history)-1] = (
                self.webcam_image.landmark_points)
        else:
            self.landmarks_history.append(self.webcam_image.landmark_points)

        landmarks_sum = []
        for i in range(len(self.webcam_image.landmark_points)):
            landmarks_sum.append([0, 0])
        for i in range(len(self.landmarks_history)):
            for j in range(len(self.landmarks_history[i])):
                landmarks_sum[j][0] += self.landmarks_history[i][j][0]
                landmarks_sum[j][1] += self.landmarks_history[i][j][1]

        for i in range(len(landmarks_sum)):
            self.webcam_image.landmark_points[i] = (
                int(landmarks_sum[i][0] / len(self.landmarks_history)),
                int(landmarks_sum[i][1] / len(self.landmarks_history)))

    def get_framerate(self):
        framerate = self.webcam.get(cv2.CAP_PROP_FPS)
        if framerate == 0:
            return 60
        return framerate

    def update_face_swap(self):
        ret, frame = self.webcam.read()
        self.swapped_face = np.zeros(frame.shape[:3], np.uint8)
        self.webcam_image.image = frame.copy()

        if(self.flip_camera):
            self.webcam_image.image = cv2.flip(self.webcam_image.image, 1)

        if(self.paused):
            return

        try:
            if not ret: raise RuntimeError("Could not read frame correctly")

            self.webcam_image_prev = self.webcam_image.copy()

            self.webcam_image.detect_first_face(self.frontal_face_detector)
            self.webcam_image.detect_landmark_points(self.facial_landmark_predictor)
            self.webcam_image.calculate_landmark_point_indices()
            self.webcam_image.calculate_head_tilt

            # webcam_first_face = FaceSwapHelper.detect_face_via_haarcascade(
            #     frame, self.face_cascade_classifier)        

            # self.webcam_landmark_points_prev = self.webcam_landmark_points
            # self.webcam_landmark_points = FaceSwapHelper.detect_facial_landmarks(
            #   frame, webcam_first_face, self.facial_landmark_predictor)  
            # if len(self.webcam_landmark_points) == 0: 
            #   raise RuntimeError("No landmarks detected")

            self.smooth_landmark_movement()
            
            self.swapped_face = FaceSwapHelper.swap_landmark_point_triangles_with_tilt(
                self.source_image, self.source_image_tilted, 
                self.webcam_image, self.swapped_face)
            self.swapped_face = FaceSwapHelper.seamless_clone_swapped_face(
                self.webcam_image.image, self.webcam_image.landmark_points,
                self.swapped_face)

            self.swapped_face = FaceSwapHelper.replace_eyes_mouth(
                self.swapped_face, self.webcam_image.image, 
                self.webcam_image.landmark_points)

            # FaceSwapHelper.draw_facial_landmarks(
            #   self.swapped_face, self.webcam_landmark_points)

            self.successful = True
        except RuntimeError as e: 
            self.successful = False
            self.swap_with_previous_landmark_points()
        except AttributeError as e:
            self.successful = False
    
    def swap_with_previous_landmark_points(self):
        if (self.webcam_image_prev.landmark_points == 0 or 
            self.webcam_image_prev.landmark_points is None):
            return
        self.swapped_face = FaceSwapHelper.swap_landmark_point_triangles(
            self.source_image, self.webcam_image, self.swapped_face)
        self.swapped_face = FaceSwapHelper.seamless_clone_swapped_face(
            self.webcam_image.image, self.webcam_image_prev.landmark_points, 
            self.swapped_face)

    def save_swapped_face(self, path):
        cv2.imwrite(path, self.swapped_face)