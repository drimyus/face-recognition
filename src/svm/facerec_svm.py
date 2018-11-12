from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
import cv2
import sys
import os

from src.face import FaceDet


class FaceRecSvm:
    def __init__(self):

        self.detector = FaceDet(detect_mode="dlib5")

        # location of classifier model
        cur = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(cur, os.pardir))
        root = os.path.abspath(os.path.join(_root, os.pardir))
        model_dir = os.path.join(root, "model/classifier")
        self.model_path = model_dir + "/dlib5/model.pkl"

        # location of dataset
        self.dataset = os.path.join(root, "dataset/People")

        # load the model
        sys.stdout.write("Loading the model.\n")
        success = self.load()

        if not success:
            # No exist trained model, so training...
            self.model = self.train()

    def load(self):
        if os.path.isfile(self.model_path):
            try:
                # loading
                self.model = joblib.load(self.model_path)
                return True
            except Exception as ex:
                print(ex)
        else:
            sys.stdout.write("    No exist Model {}, so will train\n".format(self.model_path))

    def train(self):
        sys.stdout.write("Training the model.\n")

        if not os.path.isdir(self.dataset):
            sys.stderr.write("No exist Dataset {}\n".format(self.dataset))
            exit(1)

        # initialize the data matrix and labels list
        data = []
        labels = []

        """-----------------------------------------------------------------------------------------"""
        sys.stdout.write("Scanning the dataset.\n")
        # loop over the input images
        for dirname, dirnames, filenames in os.walk(self.dataset):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)

                for filename in os.listdir(subject_path):

                    sys.stdout.write("\r    scanning: {} {}".format(subject_path, filename))
                    sys.stdout.flush()

                    img = cv2.imread(os.path.join(subject_path, filename))
                    rects, descriptions = self.detector.descript(img)
                    if len(rects) == 0:
                        continue
                    # get label, histogram
                    label, hist = subdirname, descriptions[0]

                    data.append(hist)
                    labels.append(label)

        """-----------------------------------------------------------------------------------------"""
        sys.stdout.write("\nConfigure the SVM model.\n")
        # Configure the model : linear SVM model with probability capabilities
        model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                    tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                    decision_function_shape='ovr', random_state=None)
        # model = SVC(probability=True)

        # Train the model
        model.fit(data, labels)

        joblib.dump(model, self.model_path)

        """-----------------------------------------------------------------------------------------"""
        sys.stdout.write("Finished the training.\n")
        return model

    def recog(self, description):

        description = description.reshape(1, -1)

        # Get a prediction from the model including probability:
        probab = self.model.predict_proba(description)

        propval = np.max(probab)
        propind = np.argmax(probab)

        # Rearrange by size
        propsort = np.sort(probab, axis=None)[::-1]

        if propsort[0] / propsort[1] < 1.5:
            predlbl = "UnKnown"
            propval = 0.
        else:
            predlbl = self.model.classes_[propind]

        # print(predlbl, propind, probab)
        return predlbl, propval
