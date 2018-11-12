import sys

from .face_mode.face_dlib_68 import FaceDlib68
from .face_mode.face_dlib_5 import FaceDlib5
from .face_mode.face_haar import FaceHaar

# ["dlib5", "dlib68", "haar"]


def FaceDet(detect_mode="dlib68"):

    if detect_mode == "dlib68":
        return FaceDlib68()
    elif detect_mode == "dlib5":
        return FaceDlib5()
    elif detect_mode == "haar":
        return FaceHaar()
    else:
        sys.stderr.write("there is no such mode. it should be one of ['dlib68', 'dlib5', 'haar']\n")
        sys.exit(1)
