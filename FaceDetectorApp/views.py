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

def img_capture(frames_raw, frames_nhw):
    video = cv2.VideoCapture(0)
    try:
        while True:
            (grabbed, frame) = video.read()
            frames_raw.put(cv2.flip(frame, 1))
            frame = frame.transpose((2, 0, 1))
            frames_nhw.put(frame)
    except KeyboardInterrupt:
        video.close()
        exit()

def ftr_extract(extractor, frames_nhw, extrac_rslt):
    extractor.cuda()
    try:
        while True:
            with torch.no_grad():
                frame = frames_nhw.get()
                size = frame.shape[1:]
                img = preprocess(frame)
                img = at.totensor(img[None]).float()
                scale = img.shape[3] / size[1]
                h = extractor(img)
                extrac_rslt.put(h)
    except KeyboardInterrupt:
        exit()

def rgn_proposal(rpn, extrac_rslt, rpn_rslt):
    rpn.cuda()
    try:
        while True:
            with torch.no_grad():
                h = extrac_rslt.get()
                (rpn_locs, rpn_scores, rois, roi_indices, anchor) = rpn(h, (600, 800), 1.25)
                rpn_rslt.put((h, rois, roi_indices))
    except KeyboardInterrupt:
        exit()

def fast_rcnn(roi_head, rpn_rslt, roihead_rslt):
    roi_head.cuda()
    try:
        while True:
            with torch.no_grad():
                h, rois, roi_indices = rpn_rslt.get()
                roi_cls_locs, roi_scores = roi_head(h, rois, roi_indices)
                roihead_rslt.put((rois, roi_cls_locs, roi_scores))
    except KeyboardInterrupt:
        exit()

def lbl_box(roihead_rslt, frames_raw, frames_ready):
    try:
        while True:
            rois, roi_cls_locs, roi_scores = roihead_rslt.get()
            pred_bboxes, pred_labels, pred_scores = recover(roi_cls_locs, roi_scores, rois, 1.25, (600, 800))

            pred_bboxes = np.array(pred_bboxes).reshape((-1, 4))
            pred_scores = np.array(pred_scores, ).ravel()

            mask = np.where(pred_scores > 0.8)
            pred_bboxes = pred_bboxes[mask]
            pred_scores = pred_scores[mask]

            frame_raw = frames_raw.get()
            for ((ymin, xmin, ymax, xmax), score) in zip(pred_bboxes, pred_scores):
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                cv2.rectangle(frame_raw, (xmin, ymin), (xmax, ymax), (0, 225, 0), 2)
                cv2.putText(frame_raw, str(score), (xmin - 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),2)
            frames_ready.put(frame_raw)
    except KeyboardInterrupt:
        exit()


from multiprocessing import Queue
from .tasks import img_capture, ftr_extract, rgn_proposal, fast_rcnn, lbl_box

class FaceDetector():
    qsize = 1
    def __init__(self):
        self.frames_raw = Queue(maxsize=FaceDetector.qsize)
        self.frames_nhw = Queue(maxsize=FaceDetector.qsize)
        self.frames_ready = Queue(maxsize=FaceDetector.qsize)

        self.extrac_rslt = Queue(maxsize=FaceDetector.qsize)
        self.rpn_rslt = Queue(maxsize=FaceDetector.qsize)
        self.roihead_rslt = Queue(maxsize=FaceDetector.qsize)

        faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
        pretrain_path = "D:/FRCNN_DATA/Faster-R-CNN-Pytorch-models/usable/fasterrcnn_04142205_0.49527486293443446"
        state = torch.load(pretrain_path)
        pretrained_model = state["model"]
        faster_rcnn.load_state_dict(pretrained_model)

        img_capture.delay(self.frames_raw, self.frames_nhw)
        # ftr_extract.delay(self.extractor, self.frames_nhw, self.extrac_rslt)
        # rgn_proposal.delay(self.rpn, self.extrac_rslt, self.rpn_rslt)
        # fast_rcnn.delay(self.roi_head, self.rpn_rslt, self.roihead_rslt)
        # lbl_box.delay(self.roihead_rslt, self.frames_raw, self.frames_ready)

        # self.extractor, self.rpn, self.roi_head = faster_rcnn.extractor, faster_rcnn.rpn, faster_rcnn.head
        #
        # self.cpt_proc = Process(target=img_capture,
        #                         args=(self.frames_raw, self.frames_nhw))
        #
        # self.xtrc_proc = Process(target=ftr_extract,
        #                          args=(self.extractor, self.frames_nhw, self.extrac_rslt))
        #
        # self.rpn_proc = Process(target=rgn_proposal,
        #                         args=(self.rpn, self.extrac_rslt, self.rpn_rslt))
        #
        # self.roihead_proc = Process(target=fast_rcnn,
        #                             args=(self.roi_head, self.rpn_rslt, self.roihead_rslt))
        #
        # self.lbox_proc = Process(target=lbl_box,
        #                          args=(self.roihead_rslt, self.frames_raw, self.frames_ready))
        #
        # self.cpt_proc.start()
        # self.xtrc_proc.start()
        # self.rpn_proc.start()
        # self.roihead_proc.start()
        # self.lbox_proc.start()

        # self.pool = Pool(processes=5)
        # self.pool.apply(img_capture, args=(self.frames_raw, self.frames_nhw))
        # self.pool.apply(ftr_extract, args=(self.extractor, self.frames_nhw, self.extrac_rslt))
        # self.pool.apply(rgn_proposal, args=(self.rpn, self.extrac_rslt, self.rpn_rslt))
        # self.pool.apply(fast_rcnn, args=(self.roi_head, self.rpn_rslt, self.roihead_rslt))
        # self.pool.apply(lbl_box, args=(self.roihead_rslt, self.frames_raw, self.frames_ready))

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