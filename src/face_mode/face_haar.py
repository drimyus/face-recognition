import cv2
import numpy as np
import os
import sys
import math


class FaceHaar:
    def __init__(self):

        # location of detector model
        cur = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(cur, os.pardir))
        root = os.path.abspath(os.path.join(_root, os.pardir))
        detector_dir = os.path.join(root, "model/detector")

        self.face_w, self.face_h = 130, 151

        # init the opencv's cascade haarcascade
        detector_path = os.path.join(detector_dir, "haarcascade_frontalface_alt2.xml")
        if not os.path.isfile(detector_path):
            sys.stderr.write("no exist detector.\n")
            sys.exit(1)

        self.detector = cv2.CascadeClassifier(detector_path)
        self.predictor = None

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        rects = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30),
                                               maxSize=(110, 110))
        # crop the image for the faces
        faces = []
        for (x, y, w, h) in rects:
            face = gray[max(0, y): min(y + h, height), max(0, x):min(x + w, width)]
            face = cv2.resize(face, (self.face_w, self.face_h), cv2.INTER_CUBIC)
            faces.append(face)
        return rects, faces
