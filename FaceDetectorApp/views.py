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

from App.face_detection_model_splited_test import recover
from data.dataset import preprocess
from utils import array_tool as at

from queue import Queue

class FaceDetector():
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.frames_raw = Queue(maxsize=1)
        self.frames_nhw = Queue(maxsize=1)
        self.frames_ready = Queue(maxsize=1)

        self.extrac_rslt = Queue(maxsize=1)
        self.rpn_rslt = Queue(maxsize=1)
        self.roihead_rslt = Queue(maxsize=1)

        faster_rcnn = FasterRCNNVGG16(n_fg_class=1).cuda()
        pretrain_path = "D:/FRCNN_DATA/Faster-R-CNN-Pytorch-models/usable/fasterrcnn_04142205_0.49527486293443446"
        state = torch.load(pretrain_path)
        pretrained_model = state["model"]
        faster_rcnn.load_state_dict(pretrained_model)

        self.extractor, self.rpn, self.roi_head = faster_rcnn.extractor, faster_rcnn.rpn, faster_rcnn.head

        threading.Thread(target=self.img_capture, args=()).start()
        threading.Thread(target=self.ftr_extract, args=()).start()
        threading.Thread(target=self.rgn_proposal, args=()).start()
        threading.Thread(target=self.fast_rcnn, args=()).start()
        threading.Thread(target=self.lbl_box, args=()).start()

    def __del__(self):
        self.video.release()

    def img_capture(self):
        while True:
            with torch.no_grad():
                (grabbed, frame) = self.video.read()
                self.frames_raw.put(cv2.flip(frame, 1))
                frame = frame.transpose((2, 0, 1))
                self.frames_nhw.put(frame)

    def ftr_extract(self):
        while True:
            with torch.no_grad():
                frame = self.frames_nhw.get()
                size = frame.shape[1:]
                img = preprocess(frame)
                img = at.totensor(img[None]).float()
                scale = img.shape[3] / size[1]
                h = self.extractor(img)
                self.extrac_rslt.put(h)

    def rgn_proposal(self):
        while True:
            with torch.no_grad():
                h = self.extrac_rslt.get()
                (rpn_locs, rpn_scores, rois, roi_indices, anchor) = self.rpn(h, (600, 800), 1.25)
                self.rpn_rslt.put((h, rois, roi_indices))

    def fast_rcnn(self):
        while True:
            with torch.no_grad():
                h, rois, roi_indices = self.rpn_rslt.get()
                roi_cls_locs, roi_scores = self.roi_head(h, rois, roi_indices)
                self.roihead_rslt.put((rois, roi_cls_locs, roi_scores))

    def lbl_box(self):
        while True:
            rois, roi_cls_locs, roi_scores = self.roihead_rslt.get()
            pred_bboxes, pred_labels, pred_scores = recover(roi_cls_locs, roi_scores, rois, 1.25, (600, 800))

            pred_bboxes = np.array(pred_bboxes).reshape((-1, 4))
            pred_scores = np.array(pred_scores, ).ravel()

            mask = np.where(pred_scores > 0.8)
            pred_bboxes = pred_bboxes[mask]
            pred_scores = pred_scores[mask]

            frame_raw = self.frames_raw.get()
            for ((ymin, xmin, ymax, xmax), score) in zip(pred_bboxes, pred_scores):
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                cv2.rectangle(frame_raw, (xmin, ymin), (xmax, ymax), (0, 225, 0), 2)
                cv2.putText(frame_raw, str(score), (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),2)
            self.frames_ready.put(frame_raw)

    def get_frame(self):
        image = self.frames_ready.get()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


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