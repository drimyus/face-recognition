import cv2
import numpy as np
import dlib
import os
import sys
import math

# dlib's landmarks
JAW_POINTS = list(range(0, 17))
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NUM_TOTAL_POINTS = 68


def pose_roll_deg(landmarks):
    right_eye = [0.0, 0.0]
    left_eye = [0.0, 0.0]

    # Eye: Fine the center of the right eye by averaging the points
    for i in RIGHT_EYE_POINTS:
        right_eye[0] += landmarks[i][0] / len(RIGHT_EYE_POINTS)
        right_eye[1] += landmarks[i][1] / len(RIGHT_EYE_POINTS)
    # Find the center of the left eye by averaging the points
    for i in LEFT_EYE_POINTS:
        left_eye[0] += landmarks[i][0] / len(LEFT_EYE_POINTS)
        left_eye[1] += landmarks[i][1] / len(LEFT_EYE_POINTS)

    # Calculate the roll with eye position
    roll_eye = - math.atan((left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0])) * 180 / math.pi

    # Calculate the roll with noise position
    roll_nose = math.atan((landmarks[8][0] - landmarks[27][0]) / (landmarks[8][1] - landmarks[27][1])) * 180 / math.pi

    return (roll_eye + roll_nose) / 2


def rotatePoint_with_center(cen_pt, pt, rad):
    trans_pt = (pt[0] - cen_pt[0], pt[1] - cen_pt[1])
    rot_pt = rotatePoint(trans_pt, rad)
    res_pt = (int(rot_pt[0] + cen_pt[0]), int(rot_pt[1] + cen_pt[1]))
    return res_pt


def rotatePoint(pt, rad):
    px = pt[0]
    py = pt[1]
    x = np.cos(rad) * px - np.sin(rad) * py
    y = np.sin(rad) * px + np.cos(rad) * py
    rot_pt = (x, y)
    return rot_pt


class FaceDlib68:
    def __init__(self, profile_thresh=0.7, margin_top=0.6, margin_side=0.1):

        # location of detector model
        cur = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(cur, os.pardir))
        root = os.path.abspath(os.path.join(_root, os.pardir))
        detector_dir = os.path.join(root, "model/detector")

        self.profile_thresh = profile_thresh
        self.margin_top = margin_top
        self.margin_side = margin_side

        self.face_w, self.face_h = 130, 151

        # init the dlib's face detector
        self.detector = dlib.get_frontal_face_detector()

        # init face shape predictor
        shape_predictor_path = os.path.join(detector_dir, "shape_predictor_68_face_landmarks.dat")
        if not os.path.isfile(shape_predictor_path):
            sys.stderr.write("no exist shape predictor.\n")
            sys.exit(1)
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

        # init the face descriptor
        recognizer_path = os.path.join(detector_dir, "dlib_face_recognition_resnet_model_v1.dat")
        if not os.path.isfile(recognizer_path):
            sys.stderr.write("no exist shape predictor.\n")
            sys.exit(1)
        self.recognizer = dlib.face_recognition_model_v1(recognizer_path)

    def calc_opt_crop_border(self, landmarks):
        roll_deg = pose_roll_deg(landmarks)
        roll_rad = np.deg2rad(roll_deg)

        # center of rotation face rect
        (cen_x, cen_y) = landmarks[33][0], landmarks[33][1]

        # calculate the new landmarks rotated with roll(radian)
        new_landmarks = []
        for pt in landmarks:
            new_pt = rotatePoint_with_center(cen_pt=(cen_x, cen_y), pt=pt, rad=roll_rad)
            new_landmarks.append(new_pt)

        zip_pts = list(zip(*new_landmarks))
        # corner of the rotated landmarks
        (new_left, new_top, new_right, new_bottom) = (
        min(zip_pts[0]), min(zip_pts[1]), max(zip_pts[0]), max(zip_pts[1]))
        (x, y, w, h) = (new_left, new_top, new_right - new_left, new_bottom - new_top)

        # expend with margin
        (x, y, w, h) = (int(x - self.margin_side / 2 * w),
                        int(y - self.margin_top * h),
                        int(w * (self.margin_side + 1)),
                        int(h * (self.margin_top + 1)))

        # back to the rotate with roll
        origin_corners = []
        for new_corner in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            ori_corner = rotatePoint_with_center((cen_x, cen_y), new_corner, -roll_rad)
            origin_corners.append(ori_corner)

        zip_pts = list(zip(*origin_corners))
        # corner of the rotated landmarks
        (ori_left, ori_top, ori_right, ori_bottom) = (min(zip_pts[0]), min(zip_pts[1]), max(zip_pts[0]), max(zip_pts[1]))
        (ori_x, ori_y, ori_w, ori_h) = (ori_left, ori_top, ori_right - ori_left, ori_bottom - ori_top)

        new_rect = (x, y, w, h)
        origin_rect = (ori_x, ori_y, ori_w, ori_h)
        center_pt = (cen_x, cen_y)

        return {'new_rect': new_rect,
                'origin_rect': origin_rect,
                'roll_deg': roll_deg,
                'center_pt': center_pt}

    def align_crop_Face(self, crop_coefs, frame):

        (x, y, w, h) = crop_coefs['new_rect']
        (ori_x, ori_y, ori_w, ori_h) = crop_coefs['origin_rect']
        roll_deg = crop_coefs['roll_deg']
        # (cen_x, cen_y) = crop_coefs['center_pt']

        ori_face = frame[ori_y:ori_y + ori_h, ori_x:ori_x + ori_w]
        # cv2.imshow("ori", ori_face)

        """   alignment and crop  """
        # rotate the image by roll degree
        (cen_x, cen_y) = (ori_w // 2, ori_h // 2)
        M = cv2.getRotationMatrix2D((cen_x, cen_y), -roll_deg, 1.0)

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (w / 2) - cen_x
        M[1, 2] += (h / 2) - cen_y
        align_face = cv2.warpAffine(ori_face, M, (w, h))
        # cv2.imshow("align", align_face)

        resize_face = cv2.resize(align_face, (self.face_w, self.face_h))
        return resize_face, crop_coefs['origin_rect']

    def check_frontal_face(self, landmarks):
        # only for the frontal faces
        left = abs(landmarks[39][0] - landmarks[0][0])
        right = abs(landmarks[42][0] - landmarks[16][0])
        if right != 0 and left != 0:
            if float(left) / right > self.profile_thresh and float(right) / left > self.profile_thresh:
                return True
        return False

    def _detect_face(self, frame):
        rects = self.detector(frame, 0)
        return rects

    def _recognize_face(self, frame, rect):
        shape = self.shape_predictor(frame, rect)
        face_description = self.recognizer.compute_face_descriptor(frame, shape)
        return np.asarray(face_description)

    # for training with the whole dataset
    def descript(self, frame):

        dets = self._detect_face(frame)

        descriptions = []
        rects = []
        # align and crop the detected faces
        for k, r in enumerate(dets):
            # get the landmarks/parts for the face in rect r
            face_description = self._recognize_face(frame, r)

            descriptions.append(np.asarray(face_description))
            rects.append((r.left(), r.top(), r.right(), r.bottom()))

        return rects, descriptions


if __name__ == '__main__':

    fa = FaceDlib68()
    image = cv2.imread("../../dataset/People/Anders/DSC_2773.jpg")
    image = cv2.resize(image, (400, 800))
    fa._recognize_face(image)

