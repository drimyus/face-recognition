import sys
import time
import cv2
import dlib

from src.face import FaceDet
from src.svm.facerec_svm import FaceRecSvm


class LFT_GW_SVM:

    def __init__(self):
        # Initialize capture from video
        self.cap = None

        # init the recognizer model
        self.detector = FaceDet(detect_mode='dlib5')
        # init the recognizer model
        self.classifier = FaceRecSvm()

    def run(self, video_path, skips=10):

        sys.stdout.write("\nrunning\n")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.stderr.write("cannot open such video file {}\n".format(video_path))
            sys.exit(1)

        """ ---------- Parameters for tackers --------------------------------------------------"""
        rect_color = (0, 255, 0)
        text_color = (255, 255, 0)
        tracking_quality_thresh = 7

        faceTrackers = {}

        margin = 10

        start = 0  # for estiamte the frame rate
        frameCounter = -1  # for estiamte the frame rate
        unknown_person_id = 1
        """ ---------- Main Loop for face recognition --------------------------------------------------"""
        try:
            # Loop to recognize faces
            while True:
                # capture frame
                ret, frame = self.cap.read()
                frameCounter += 1
                if not ret:
                    sys.stderr.write("cannot read frame.\n")
                    break

                """ ------------------------------------------------------------------------------ """
                # check the tracking quality status of faceTracker
                # if so, delete its faceTracker
                fidsToDelete = []
                for fid in faceTrackers.keys():
                    tracking_quality = faceTrackers[fid].update(frame)
                    if tracking_quality < tracking_quality_thresh:
                        fidsToDelete.append(fid)

                for fid in fidsToDelete:
                    print("removing fid" + str(fid) + "from the list of Trackers")
                    faceTrackers.pop(fid, None)

                # detect the faces from the new frame
                if (frameCounter % skips) == 0:

                    interval = time.time() - start
                    print("\restimated fps: {:3.2f}".format(skips * 1 / interval))
                    # sys.stdout.flush()
                    start = time.time()

                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector._detect_face(frame)

                    for rect in faces:
                        matched_fid = None

                        (x, y, w, h) = (int(rect.left()), int(rect.top()), int(rect.width()), int(rect.height()))
                        x_bar = x + 0.5 * w
                        y_bar = y + 0.5 * h

                        # find the nearest face among the tracking faces
                        for fid in faceTrackers.keys():
                            tracked_position = faceTrackers[fid].get_position()

                            t_x = int(tracked_position.left())
                            t_y = int(tracked_position.top())
                            t_w = int(tracked_position.width())
                            t_h = int(tracked_position.height())

                            t_x_bar = t_x + 0.5 * t_w
                            t_y_bar = t_y + 0.5 * t_h

                            if ((t_x <= x_bar <= (t_x + t_w)) and
                                    (t_y <= y_bar <= (t_y + t_h)) and
                                    (x <= t_x_bar <= (x + w)) and
                                    (y <= t_y_bar <= (y + h))):
                                matched_fid = fid

                        # there is no similar face, create new tracker for it
                        if matched_fid is None:
                            description = self.detector._recognize_face(frame, rect)
                            new_fid, _ = self.classifier.recog(description)

                            if new_fid is 'UnKnown':
                                new_fid = new_fid + "_" + str(unknown_person_id)
                                unknown_person_id += 1

                            print("Creating new tracker for" + str(new_fid))

                            # create new tracker
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(frame,
                                                dlib.rectangle(x - margin, y - margin, x + w + margin, y + h + margin))
                            faceTrackers[new_fid] = tracker

                # show the detected faces with its fid(face name)
                for fid in faceTrackers.keys():
                    tracked_position = faceTrackers[fid].get_position()
                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())

                    cv2.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), rect_color, 1)

                    cv2.putText(frame, fid, (int(t_x), int(t_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                cv2.imshow("Face Tracking", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 100)

        except KeyboardInterrupt as ex:
            print(ex)
            pass

        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == '__main__':
    fr = LFT_GW_SVM()
    fr.run(video_path="../../videos/my_video-6.mkv", skips=20)
