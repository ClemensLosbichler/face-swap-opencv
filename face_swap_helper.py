import cv2
import dlib
import numpy as np

class FaceSwapHelper:
    @staticmethod
    def detect_facial_landmarks(image, face, facial_landmark_predictor):
        landmark_points = facial_landmark_predictor(image, face)
        landmark_point_array = []
        for i in range(0, 68):
            x = landmark_points.part(i).x
            y = landmark_points.part(i).y
            landmark_point_array.append((x, y))
        return landmark_point_array

    @staticmethod
    def detect_first_face(image, frontal_face_detector):
        faces = frontal_face_detector(image)
        for face in faces:
            face = dlib.rectangle(int(faces[0].left()), int(faces[0].top()),
            int(faces[0].right()), int(faces[0].bottom()))
            return face
        return None

    @staticmethod
    def detect_facial_landmarks_triangulated(image, frontal_face_detector, face_landmark_predictor):
        first_face = FaceSwapHelper.detect_first_face(image, frontal_face_detector)
        landmark_points = FaceSwapHelper.detect_facial_landmarks(
            image, first_face, face_landmark_predictor)  
        triangles = FaceSwapHelper.triangulate_landmark_points(landmark_points)
        triangle_indices = FaceSwapHelper.index_landmark_point_triangles(landmark_points, triangles)
        return (landmark_points, triangle_indices)

    @staticmethod
    def triangulate_landmark_points(landmark_points):
        convexhull = cv2.convexHull(np.array(landmark_points, dtype=np.int32))
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmark_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        return triangles

    @staticmethod
    def detect_face_via_dlib(image, frontal_face_detector):
        webcam_first_face = FaceSwapHelper.detect_first_face(np.array(image), frontal_face_detector)
        if webcam_first_face == 0: 
            raise RuntimeError("No face detected")
        return webcam_first_face

    @staticmethod
    def detect_face_via_haarcascade(image, face_cascade_classifier):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          
        faces = face_cascade_classifier.detectMultiScale(gray, scaleFactor = 1.1,
            minNeighbors = 5, minSize = (10,10))

        for(x, y, w, h) in faces:
            return dlib.rectangle(x, y, x+w, y+h)
        raise RuntimeError("No face detected")

    @staticmethod
    def index_landmark_point_triangles(landmark_points, landmark_point_triangles):
        landmark_points_array = np.array(landmark_points, np.int32)
        triangle_indices = []
        for t in landmark_point_triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            index_pt1 = np.where((landmark_points_array == pt1).all(axis=1))
            index_pt1 = index_pt1[0]
            index_pt2 = np.where((landmark_points_array == pt2).all(axis=1))
            index_pt2 = index_pt2[0]
            index_pt3 = np.where((landmark_points_array == pt3).all(axis=1))
            index_pt3 = index_pt3[0]
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                triangle_indices.append(triangle)
        return triangle_indices

    @staticmethod
    def get_triangle_points(image, landmark_points, triangle_index):
        triangle_points_absolute = FaceSwapHelper.get_landmark_point_triangle_from_index(
            landmark_points, triangle_index)
        (x, y, w, h) = cv2.boundingRect(triangle_points_absolute)
        image_triangle = image[y:y+h, x:x+w]

        triangle_points_relative = np.array([
            [triangle_points_absolute[0][0] - x, triangle_points_absolute[0][1] - y],
            [triangle_points_absolute[1][0] - x, triangle_points_absolute[1][1] - y],
            [triangle_points_absolute[2][0] - x, triangle_points_absolute[2][1] - y]], np.int32)

        return triangle_points_relative, triangle_points_absolute, image_triangle

    @staticmethod
    def get_landmark_point_triangle_from_index(landmark_points, index):
        landmark_points_array = np.array(landmark_points)
        triangle_point1 = landmark_points_array[index[0]][0]
        triangle_point2 = landmark_points_array[index[1]][0]
        triangle_point3 = landmark_points_array[index[2]][0]
        triangle = np.array([triangle_point1, triangle_point2, triangle_point3], np.int32)
        return triangle

    def get_landmark_point_triangle_warp_mask(triangle):
        (x, y, w, h) = cv2.boundingRect(triangle)
        mask = np.zeros((h, w), np.uint8)
        points = np.array([[triangle[0][0] - x, triangle[0][1] - y],
                            [triangle[1][0] - x, triangle[1][1] - y],
                            [triangle[2][0] - x, triangle[2][1] - y]], np.int32)
        cv2.fillConvexPoly(mask, points, 255)
        return mask

    @staticmethod
    def draw_facial_landmarks(img, landmark_points, color=(255, 255, 255), radius=3):
        for p in landmark_points:
            cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)

    @staticmethod
    def draw_facial_landmark_triangles(img, triangle):
        cv2.line(img, (triangle[0][0], triangle[0][1]), 
            (triangle[1][0], triangle[1][1]), (255, 255, 255))
        cv2.line(img, (triangle[1][0], triangle[1][1]), 
            (triangle[2][0], triangle[2][1]), (255, 255, 255))
        cv2.line(img, (triangle[2][0], triangle[2][1]), 
            (triangle[0][0], triangle[0][1]), (255, 255, 255))

    @staticmethod
    def get_head_tilt_from_landmarks(landmark_points):
        eye1_width = landmark_points[36][0] - landmark_points[0][0]
        eye2_width = landmark_points[16][0] - landmark_points[45][0]

        if (eye1_width > eye2_width):
            return 1        # right
        return -1         # left

    @staticmethod
    def seamless_clone_swapped_face(destination_image, destination_image_landmark_points, swapped_face):
        destination_image_face_hull = cv2.convexHull(np.array(destination_image_landmark_points))

        destination_image_grayscale = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
        destination_image_face_mask = np.zeros_like(destination_image_grayscale)
        destination_image_face_mask = cv2.fillConvexPoly(
            destination_image_face_mask, destination_image_face_hull, 255)

        (x, y, w, h) = cv2.boundingRect(destination_image_face_hull)
        center = (int((x + x + w) / 2), int((y + y + h) / 2))
        swapped_face = cv2.seamlessClone(
            swapped_face, destination_image, destination_image_face_mask, center, cv2.NORMAL_CLONE)
        return swapped_face

    @staticmethod
    def swap_landmark_point_triangles_with_tilt(source_image, source_image_tilted, 
        destination_image, swapped_face):
        if destination_image.head_tilt != source_image.head_tilt:
            return FaceSwapHelper.swap_landmark_point_triangles(
                source_image_tilted, destination_image, swapped_face)
        else:
            return FaceSwapHelper.swap_landmark_point_triangles(
                source_image, destination_image, swapped_face)

    @staticmethod
    def swap_landmark_point_triangles(source_image, destination_image, swapped_face):
        for triangle_index in source_image.landmark_point_indices:
            (source_triangle_points_relative, 
            source_triangle_points_absolute, 
            source_triangle) = FaceSwapHelper.get_triangle_points(
                source_image.image, source_image.landmark_points, triangle_index)

            (destination_triangle_points_relative, 
            destination_triangle_points_absolute, 
            destination_triangle) = FaceSwapHelper.get_triangle_points(
                destination_image.image, destination_image.landmark_points, triangle_index)

            (x, y, w, h) = cv2.boundingRect(destination_triangle_points_absolute)
            if x < 0 or y < 0 or w < 0 or h < 0: 
                raise ValueError(
                    "Triangle Bounding Rect for Index " + str(triangle_index) + " is negative")

            warp_mask = FaceSwapHelper.get_landmark_point_triangle_warp_mask(
                destination_triangle_points_absolute)
            warp_mat = cv2.getAffineTransform(
                np.float32(source_triangle_points_relative), np.float32(destination_triangle_points_relative))
            warped_triangle = cv2.warpAffine(source_triangle, warp_mat, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=warp_mask)

            gray = cv2.cvtColor(swapped_face[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            _, mask_remove_triangles = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
            try:
                warped_triangle = cv2.bitwise_and(
                    warped_triangle, warped_triangle, mask=mask_remove_triangles)
            except Exception as e:
                raise RuntimeError(e)

            swapped_face[y:y+h, x:x+w] = cv2.add(swapped_face[y:y+h, x:x+w], warped_triangle)
        return swapped_face

    def replace_landmarks(image1, image2, landmark_points, seamless_replace = True):
        part = np.int32([landmark_points])
        mask = np.zeros((image1.shape[0], image1.shape[1]), dtype='uint8')
        cv2.fillConvexPoly(mask, part, (255, 255, 255))
        
        if not seamless_replace:
            image1[mask == 255] = image2[mask == 255]
            return image1

        normal = cv2.resize(image2, (image1.shape[0], image1.shape[1]))
        mask = cv2.resize(mask, (image1.shape[0], image1.shape[1]))

        hull = cv2.convexHull(part)
        (x, y, w, h) = cv2.boundingRect(hull)
        center = (int((x + x + w) / 2), int((y + y + h) / 2))
        image1 = cv2.seamlessClone(normal, image1, mask, center, cv2.NORMAL_CLONE)
        return image1

    def replace_eyes_mouth(image1, image2, landmark_points):
        image1 = FaceSwapHelper.replace_landmarks(
            image1, image2, landmark_points[60:68])
        image1 = FaceSwapHelper.replace_landmarks(
            image1, image2, landmark_points[42:48], False)
        image1 = FaceSwapHelper.replace_landmarks(
            image1, image2, landmark_points[36:42], False)
        return image1