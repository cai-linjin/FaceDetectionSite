# Create your tasks here
from __future__ import absolute_import, unicode_literals
from celery import shared_task
from FaceDetectionSite.celery import clry_app

import sys
sys.path.extend([
    # "D:\\PycharmProjects\\Faster R-CNN\\utils\\",
    # "D:\\PycharmProjects\\Faster R-CNN\\model\\",
    "D:\\PycharmProjects\\Faster R-CNN\\",
    # "D:\\PycharmProjects\\Faster R-CNN\\model\\utils",
])
import cv2
import torch
import numpy as np
from App.face_detection_model_splited_test import recover
from data.dataset import preprocess
from utils import array_tool as at

@clry_app.task
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

@clry_app.task
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

@clry_app.task
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

@clry_app.task
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

@clry_app.task
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