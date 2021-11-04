import matplotlib.pyplot as plt
from bob.io.base import load
from bob.io.base.test_utils import datafile
from bob.io.image import imshow
from bob.ip.facedetect.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

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

 
     
def main(df):

    for index, row in df.iterrows():
       
        print(index)
        print(row[0])
        file = row[0]
        print(file)

        string1 = "jpg"
        string2 = file
        if string1 in string2:
          string2 = string2.replace(string1, "png")
        print(string2)

        pil_image = Image.open("/local/scratch/datasets/CelebA/img_align_celeba_png/"+string2)
        data = image.imread("/local/scratch/datasets/CelebA/img_align_celeba_png/"+string2)

        bob_image = to_bob(data)
        image_n = bob_to_opencvbgr(bob_image)
       

        convert = cv2.cvtColor(image_n,cv2.COLOR_BGR2RGB)

        face_image = convert
        face_image *= 255 # or any coefficient
        face_image = face_image.astype(np.uint8)
        bob_image_2 = to_bob(face_image)
                
        face_eyes_norm = bob.ip.base.FaceEyesNorm(eyes_distance = 44, crop_size = (112, 112), eyes_center = (42.5, 56.5))

        leye = (row["lefteye_y"],row["lefteye_x"])
        reye = (row["righteye_y"],row["righteye_x"])
        print(leye)

        normalized_image = face_eyes_norm(bob_image_2, right_eye = leye , left_eye = reye )
        normalized_image = to_matplotlib(normalized_image)
        normalized_image = normalized_image.astype(np.uint8)
        normalized_image = cv2.cvtColor(normalized_image,cv2.COLOR_BGR2RGB)

        cv2.imwrite("/local/scratch/anushri/normalized_celeb/"+str(string2),normalized_image)
                
                

df = pd.read_csv('/home/user/anushri/LRFR/list_landmarks_align_celeba.csv')

main(df)              