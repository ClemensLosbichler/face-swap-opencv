import cv2

class WebCam:
    @staticmethod
    def count_webcams():
        for i in range(10):
            temp_camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if temp_camera.isOpened():
                temp_camera.release()
                continue
            return i