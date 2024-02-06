import cv2
import dlib
import numpy as np

from face_swap_helper import FaceSwapHelper
from face_swap_image import FaceSwapImage

IMG1_PATH = "../../Face Swap/bilder/boelzl.jpg"
IMG2_PATH = "../../Face Swap/bilder/herwig.png"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

frontal_face_detector = dlib.get_frontal_face_detector()
facial_landmark_predictor = dlib.shape_predictor(MODEL_PATH)
face_cascade_classifier = cv2.CascadeClassifier(
    "..\\OpenCV\\haarcascade_models\\haarcascade_frontalface_default.xml")

image1 = FaceSwapImage(cv2.imread(IMG1_PATH))
image2 = FaceSwapImage(cv2.imread(IMG2_PATH))

if image1.image is None or image2.image is None:
    print("Image path is wrong")
    quit()

image1.detect_first_face(frontal_face_detector)
image1.detect_landmark_points(facial_landmark_predictor)
image1.calculate_landmark_point_indices()
image1.calculate_head_tilt()
image2.detect_first_face(frontal_face_detector)
image2.detect_landmark_points(facial_landmark_predictor)
image2.calculate_landmark_point_indices()
image2.calculate_head_tilt()

if image1.head_tilt != image2.head_tilt:
    image1.image = cv2.flip(image1.image, 1)
    image1.detect_first_face(frontal_face_detector)
    image1.detect_landmark_points(facial_landmark_predictor)
    image1.calculate_landmark_point_indices()

swapped_face = np.zeros(image2.image.shape[:3], np.uint8)

swapped_face = FaceSwapHelper.swap_landmark_point_triangles(
    image1, image2, swapped_face)
swapped_face = FaceSwapHelper.seamless_clone_swapped_face(
    image2.image, image2.landmark_points, swapped_face)

cv2.imshow("Face Swap", swapped_face)

cv2.waitKey(0)
cv2.destoryAllWindows()
