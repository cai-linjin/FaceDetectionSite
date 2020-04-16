import threading
import time

from django.http.response import StreamingHttpResponse
from django.shortcuts import render

# Create your views here.

import cv2
from django.views.decorators import gzip
import numpy as np

# config sys path so that FasterRCNNVGG16 can be imported
import sys
sys.path.extend([
    # "D:\\PycharmProjects\\Faster R-CNN\\utils\\",
    # "D:\\PycharmProjects\\Faster R-CNN\\model\\",
    "D:\\PycharmProjects\\Faster R-CNN\\",
    # "D:\\PycharmProjects\\Faster R-CNN\\model\\utils",
])

from model.faster_rcnn_vgg16 import FasterRCNNVGG16
import torch

class FaceDetector():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.frame = cv2.flip(self.frame, 1)

        faster_rcnn = FasterRCNNVGG16(n_fg_class=1).cuda()
        pretrain_path = "D:/FRCNN_DATA/Faster-R-CNN-Pytorch-models/usable/fasterrcnn_04142205_0.49527486293443446"
        state = torch.load(pretrain_path)
        pretrained_model = state["model"]
        faster_rcnn.load_state_dict(pretrained_model)

        self.detector = faster_rcnn

        threading.Thread(target=self.detect, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def detect(self):
        while True:
            (self.grabbed, frame) = self.video.read()

            frame_raw = cv2.flip(frame, 1)
            frame = frame_raw.transpose((2, 0, 1))

            # detection with faster r-cnn
            pred_bboxes, pred_labels, pred_scores = self.detector.predict(torch.tensor(frame[None, :]), [[480, 640]])
            # displaying the frame
            pred_bboxes = np.array(pred_bboxes).reshape((-1, 4))
            pred_scores = np.array(pred_scores, ).ravel()

            mask = np.where(pred_scores > 0.8)
            pred_bboxes = pred_bboxes[mask]
            pred_scores = pred_scores[mask]

            for ((ymin, xmin, ymax, xmax), score) in zip(pred_bboxes, pred_scores):
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                cv2.rectangle(frame_raw, (xmin, ymin), (xmax, ymax), (0, 225, 0), 2)
                cv2.putText(frame_raw, str(score), (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),
                            2)

            self.frame = frame_raw


f_detector = FaceDetector()

def gen(detector):
    while True:
        frame = detector.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# TODO
# not loading all the time
# add some fancy decorations

@gzip.gzip_page
def livedetect(request):
    try:
        return StreamingHttpResponse(gen(f_detector), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass