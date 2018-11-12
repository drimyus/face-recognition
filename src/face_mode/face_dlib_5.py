import cv2
import numpy as np
import dlib
import os
import sys
import math

# dlib's landmarks
NOSE_POINTS = [4]
RIGHT_EYE_POINTS = list(range(2, 4))
LEFT_EYE_POINTS = list(range(0, 2))
NUM_TOTAL_POINTS = 5


class FaceDlib5:
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

        # init the dlib's face shape predictor
        detector_path = os.path.join(detector_dir, "shape_predictor_5_face_landmarks.dat")
        if not os.path.isfile(detector_path):
            sys.stderr.write("no exist shape_predictor.\n")
            sys.exit(1)
        self.shape_predictor = dlib.shape_predictor(detector_path)

        # init the face descriptor
        recognizer_path = os.path.join(detector_dir, "dlib_face_recognition_resnet_model_v1.dat")
        if not os.path.isfile(recognizer_path):
            sys.stderr.write("no exist shape predictor.\n")
            sys.exit(1)
        self.recognizer = dlib.face_recognition_model_v1(recognizer_path)

    def _detect_face(self, frame):

        rects = self.detector(frame, 0)
        return rects

    def _recognize_face(self, frame, rect):
        shape = self.shape_predictor(frame, rect)
        face_description = self.recognizer.compute_face_descriptor(frame, shape)
        return np.asarray(face_description)

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

    fa = FaceDlib5()
    image = cv2.imread("../../dataset/People/Anders/DSC_2773.jpg")
    image = cv2.resize(image, (400, 800))
    fa.descript(image)
