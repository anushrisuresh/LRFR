#imports
import matplotlib.pyplot as plt
from bob.io.base import load
from bob.io.base.test_utils import datafile
from bob.io.image import imshow
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import bob.ip.facedetect

from bob.ip.color import gray_to_rgb
import logging
import numpy as np
import pickle
import os, sys
from collections import namedtuple
import time
from bob.io.image import to_matplotlib
import pkg_resources
from bob.extension import rc
from bob.extension.download import get_file

import PIL
from PIL import Image
from bob.io.image import to_matplotlib
from bob.io.image import bob_to_opencvbgr
import cv2
from matplotlib import image
from bob.io.image import to_bob

import numpy
import os
import math
import bob.measure
import pandas as pd

#tinyface class

class TinyFacesDetector:
    def __init__(self, prob_thresh=0.5, **kwargs):
        super().__init__(**kwargs)
        import mxnet as mx

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.ip.facedetect/master/tinyface_detector.tar.gz"
        ]

        filename = get_file(
            "tinyface_detector.tar.gz",
            urls,
            cache_subdir="data/tinyface_detector",
            file_hash="f24e820b47a7440d7cdd7e0c43d4d455",
            extract=True,
        )

        self.checkpoint_path = os.path.dirname(filename)

        self.MAX_INPUT_DIM = 5000.0
        self.prob_thresh = prob_thresh
        self.nms_thresh = 0.1
        self.model_root = pkg_resources.resource_filename(
            __name__, self.checkpoint_path
        )

        sym, arg_params, aux_params = mx.model.load_checkpoint(
            os.path.join(self.checkpoint_path, "hr101"), 0
        )
        all_layers = sym.get_internals()

        meta_file = open(os.path.join(self.checkpoint_path, "meta.pkl"), "rb")
        self.clusters = pickle.load(meta_file)
        self.averageImage = pickle.load(meta_file)
        meta_file.close()
        self.clusters_h = self.clusters[:, 3] - self.clusters[:, 1] + 1
        self.clusters_w = self.clusters[:, 2] - self.clusters[:, 0] + 1
        self.normal_idx = np.where(self.clusters[:, 4] == 1)

        self.mod = mx.mod.Module(
            symbol=all_layers["fusex_output"], data_names=["data"], label_names=None
        )
        self.mod.bind(
            for_training=False,
            data_shapes=[("data", (1, 3, 224, 224))],
            label_shapes=None,
            force_rebind=False,
        )
        self.mod.set_params(
            arg_params=arg_params, aux_params=aux_params, force_init=False
        )

    @staticmethod
    def _nms(dets, prob_thresh):

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= prob_thresh)[0]

            order = order[inds + 1]
        return keep

    def detect(self, img):
        import cv2 as cv
        import mxnet as mx

        Batch = namedtuple("Batch", ["data"])

        raw_img = img
        if len(raw_img.shape) == 2:
            raw_img = gray_to_rgb(raw_img)
        assert img.shape[0] == 3, img.shape

        raw_img = to_matplotlib(raw_img)
        raw_img = raw_img[..., ::-1]

        raw_h = raw_img.shape[0]
        raw_w = raw_img.shape[1]

        raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)
        raw_img_f = raw_img.astype(np.float32)

        min_scale = min(
            np.floor(np.log2(np.max(self.clusters_w[self.normal_idx] / raw_w))),
            np.floor(np.log2(np.max(self.clusters_h[self.normal_idx] / raw_h))),
        )
        max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / self.MAX_INPUT_DIM))

        scales_down = np.arange(min_scale, 0 + 0.0001, 1.0)
        scales_up = np.arange(0.5, max_scale + 0.0001, 0.5)
        scales_pow = np.hstack((scales_down, scales_up))
        scales = np.power(2.0, scales_pow)

        start = time.time()
        bboxes = np.empty(shape=(0, 5))

        for s in scales[::-1]:
            img = cv.resize(raw_img_f, (0, 0), fx=s, fy=s)
            img = np.transpose(img, (2, 0, 1))
            img = img - self.averageImage

            tids = []
            if s <= 1.0:
                tids = list(range(4, 12))
            else:
                tids = list(range(4, 12)) + list(range(18, 25))
            ignoredTids = list(set(range(0, self.clusters.shape[0])) - set(tids))
            img_h = img.shape[1]
            img_w = img.shape[2]
            img = img[np.newaxis, :]

            self.mod.reshape(data_shapes=[("data", (1, 3, img_h, img_w))])
            self.mod.forward(Batch([mx.nd.array(img)]))
            self.mod.get_outputs()[0].wait_to_read()
            fusex_res = self.mod.get_outputs()[0]

            score_cls = mx.nd.slice_axis(
                fusex_res, axis=1, begin=0, end=25, name="score_cls"
            )
            score_reg = mx.nd.slice_axis(
                fusex_res, axis=1, begin=25, end=None, name="score_reg"
            )
            prob_cls = mx.nd.sigmoid(score_cls)

            prob_cls_np = prob_cls.asnumpy()
            prob_cls_np[0, ignoredTids, :, :] = 0.0

            _, fc, fy, fx = np.where(prob_cls_np > self.prob_thresh)

            cy = fy * 8 - 1
            cx = fx * 8 - 1
            ch = self.clusters[fc, 3] - self.clusters[fc, 1] + 1
            cw = self.clusters[fc, 2] - self.clusters[fc, 0] + 1

            Nt = self.clusters.shape[0]

            score_reg_np = score_reg.asnumpy()
            tx = score_reg_np[0, 0:Nt, :, :]
            ty = score_reg_np[0, Nt : 2 * Nt, :, :]
            tw = score_reg_np[0, 2 * Nt : 3 * Nt, :, :]
            th = score_reg_np[0, 3 * Nt : 4 * Nt, :, :]

            dcx = cw * tx[fc, fy, fx]
            dcy = ch * ty[fc, fy, fx]
            rcx = cx + dcx
            rcy = cy + dcy
            rcw = cw * np.exp(tw[fc, fy, fx])
            rch = ch * np.exp(th[fc, fy, fx])

            score_cls_np = score_cls.asnumpy()
            scores = score_cls_np[0, fc, fy, fx]

            tmp_bboxes = np.vstack(
                (rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2)
            )
            tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
            tmp_bboxes = tmp_bboxes.transpose()
            bboxes = np.vstack((bboxes, tmp_bboxes))

        refind_idx = self._nms(bboxes, self.nms_thresh)
        refind_bboxes = bboxes[refind_idx]
        refind_bboxes = refind_bboxes.astype(np.int32)

        annotations = refind_bboxes
        annots = []
        for i in range(len(refind_bboxes)):
            topleft = round(float(annotations[i][1])), round(float(annotations[i][0]))
            bottomright = (
                round(float(annotations[i][3])),
                round(float(annotations[i][2])),
            )
            width = float(annotations[i][2]) - float(annotations[i][0])
            length = float(annotations[i][3]) - float(annotations[i][1])
            right_eye = (
                round((0.37) * length + float(annotations[i][1])),
                round((0.3) * width + float(annotations[i][0])),
            )
            left_eye = (
                round((0.37) * length + float(annotations[i][1])),
                round((0.7) * width + float(annotations[i][0])),
            )
            annots.append(
                {
                    "topleft": topleft,
                    "bottomright": bottomright,
                    "reye": right_eye,
                    "leye": left_eye,
                }
            )

        return annots

def ground_truth_for_image(x,y,width,height):
	
    truth = [[x,y,width,height]]
    
    return truth

def face_detector(bob_image):
    detector = TinyFacesDetector()
    detections = detector.detect(bob_image)
    return detections

def detection_bbx_for_image(detections):
  detected_bbx = []


  for annotations in detections:
    #detection params
    topleft = annotations["topleft"]
    bottomright = annotations["bottomright"]
    #bbx params
    x1 = int(topleft[0])
    y1 = int(topleft[1])
    x2 = int(bottomright[0])
    y2 = int(bottomright[1])
    w = x2-x1
    h = y2-y1
    #img = bob_to_opencvbgr(bob_image)
    # #img_cropped = img[x1:x2, y1:y2, :]
    detected = [y1,x1,h,w]
    detected_bbx.append(detected)
    

  
  return detected_bbx

def return_eyes(detections):
    eyes=[]
    for annotations in detections:
        reye = (int(annotations["reye"][0]),int(annotations["reye"][1]))
        leye = (int(annotations["leye"][0]),int(annotations["leye"][1]))

        eyes.append((reye,leye))
    return eyes

def bounding_box(bbx):
  """Converts a bounding box (x, y, w, h) into a :py:class:`bob.ip.facedetect.BoundingBox`"""
  if isinstance(bbx, bob.ip.facedetect.BoundingBox):
    return bbx
  return bob.ip.facedetect.BoundingBox((bbx[1], bbx[0]), size = (bbx[3], bbx[2]))

def overlap(gt, det):
  gt = bounding_box(gt)
  det = bounding_box(det)

  intersection = gt.overlap(det)

  # negative size of intersection: no intersection
  if any(s <= 0 for s in intersection.size_f):
    # no overlap
    return 0.

  # compute union; reduce required overlap to the ground truth
  union = max(gt.area/4, intersection.area) + det.area - intersection.area

  # return intersection over modified union (modified Jaccard similarity)
  return intersection.area / union

overlap_threshold = 0.7

def _compare(ground_truth, detection):
  
  faces = len(ground_truth)
  if detection is None:
    return [], [], faces
  # turn into BoundingBox'es
  gt = [bounding_box(g) for g in ground_truth]
  dt = [bounding_box(d) for d in detection]
 
  # compute similarity matrix between detections and ground truth
  similarities = numpy.array([[overlap(g,d) for d in dt] for g in gt])
 
  # for each detected bounding box, find the gt with the largest overlap
  positives = []
  negatives = []
  #where = numpy.where(similarities>overlap_threshold)

  # for each detection, find the best overlap with the ground truth
  for d in range(len(dt)):
   
    if numpy.all(similarities[:,d] < overlap_threshold):
      # when no overlap is large enough: no face -> negative detection
      negatives.append(detection[d])
    else:
      # we have an overlap
      best = numpy.argmax(similarities[:,d])
      if numpy.max(similarities[best,:]) > similarities[best,d] or\
         numpy.count_nonzero(similarities[best,d:] == similarities[best,d]) > 1: # count each negative only once
        # when there is another bounding box with larger overlap with the GT -> negative detection
        # this avoids having lot of overlapping boxes
        negatives.append(detection[d])
      else:
        # Best detection with best similarity: this score is a positive
        positives.append(detection[d])
        
  return positives

def save_to_dir(compared,identity,bob_image,eyes):
    for c in compared:

        #final image dimensions
        y1 = c[0]
        x1 = c[1]
        y2 = c[2] +y1
        x2 = c[3] +x1
        image = bob_to_opencvbgr(bob_image)
        #print(x1,y1,x2,y2)
        #print(len(eyes))
        
        image_cropped = image[x1:x2, y1:y2, :]
        #print(image_cropped)
        if image_cropped.size == 0:
            #cv2.imwrite("/local/scratch/anushri/LRFR/detected_faces_tiny/"+str(identity)+".png",image_cropped)
            #cv2.imwrite("/home/user/anushri/LRFR/tester/detected_faces_tiny/"+str(identity)+".png",image_cropped)
            print("image not found for identity:"+str(identity))
            continue

        #plt.imshow(image_cropped)


        #save normalized images in folder
        for i in range(len(eyes)):
            if eyes[i][0][0]>x1 and eyes[i][0][0]<x2 and eyes[i][0][1]<y2 and eyes[i][0][1]>y1:
                
                convert = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

                face_image = convert
                #face_image *= 255 # or any coefficient
                face_image = face_image.astype(np.uint8)
                bob_image_2 = to_bob(face_image)
                
                face_eyes_norm = bob.ip.base.FaceEyesNorm(eyes_distance = 40, crop_size = (112, 112), eyes_center = (41, 63.5))

                reye = (eyes[i][0][0],eyes[i][0][1])
                leye = (eyes[i][1][0],eyes[i][1][1])

                normalized_image = face_eyes_norm(bob_image_2, right_eye = reye , left_eye = leye )
                normalized_image = to_matplotlib(normalized_image)
                normalized_image = normalized_image.astype(np.uint8)
                normalized_image = cv2.cvtColor(normalized_image,cv2.COLOR_BGR2RGB)

                #cv2.imwrite("/home/user/anushri/LRFR/tester/normalized_faces_tiny/"+str(identity)+".png",normalized_image)
                cv2.imwrite("/local/scratch/anushri/LRFR/normalized_faces_tiny_val/"+str(identity)+".png",normalized_image)
                
def main(df):

    for index,row in df.iterrows():
       
        print(str(index) + "," + str(row["FACE_ID"]))
        file = row["FILE"]

        pil_image = Image.open("/local/scratch/datasets/UCCS/validation/"+file)
        data = image.imread("/local/scratch/datasets/UCCS/validation/"+file)
        bob_image = to_bob(data)

        x = row["FACE_X"]
        y = row["FACE_Y"]
        width = row["FACE_WIDTH"]
        height = row["FACE_HEIGHT"]
        identity = row["FACE_ID"]

        ground_truth = ground_truth_for_image(x,y,width,height)
        detection = face_detector(bob_image)
        detected_bbx = detection_bbx_for_image(detection)
        #print(detected_bbx)
        eyes = return_eyes(detection)
        compared = _compare(ground_truth, detected_bbx)
        save_to_dir(compared,identity,bob_image,eyes)

df = pd.read_csv('/home/user/anushri/LRFR/validation_filtered.csv')
#df = pd.read_csv('/home/user/anushri/LRFR/training_sample.csv')


main(df)