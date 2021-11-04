from __future__ import print_function

# import required bob modules
import bob.db.atnt
import bob.io.base
import bob.io.image
import bob.ip.base
import bob.measure
import numpy, scipy.spatial
import os, sys
import numpy, math
import matplotlib
from matplotlib import pyplot
import pandas as pd
import pickle
from arcface_template import *


DISTANCE_FUNCTION = scipy.spatial.distance.cosine

train_image_directory = '/local/scratch/anushri/LRFR/LR_images_train'
val_image_directory = '/local/scratch/anushri/LRFR/LR_images_val'

db_train = "/home/user/anushri/LRFR/training_filtered.csv"
df_train = pd.read_csv(db_train)

db_val = "/home/user/anushri/LRFR/validation_filtered.csv"
df_val = pd.read_csv(db_val)

extractor = ArcFaceInsightFace_LResNet100()
extractor.use_gpu = True

model_ids = df_train.SUBJECT_ID

models = {}
probes = {}

model_count = 0


for model_id in model_ids:

    #print(df[df["SUBJECT_ID"]== model_id].FACE_ID.values)
    model_faces = df_train[df_train["SUBJECT_ID"]== model_id].FACE_ID.values

    faces = []

    for model_face in model_faces:
        
        if os.path.isfile(train_image_directory + "/"+ str(model_face)+".png"):
            image =  [bob.io.base.load(train_image_directory + "/"+ str(model_face)+".png")]
            faces.append(extractor.transform(image))

    mean_face = numpy.mean(faces, axis=0)
    models[model_id] = mean_face

    model_count +=1
    
    print("model:"+ str(model_count) + "/" + str(len(model_ids)))

probe_faces = df_val.FACE_ID

probe_count = 0 

for probe_face in probe_faces:
    if os.path.isfile(val_image_directory + "/"+ str(probe_face)+".png"):
        image =  [bob.io.base.load(val_image_directory + "/"+ str(probe_face)+".png")]
        probes[probe_face] = extractor.transform(image)

    probe_count +=1
    print("probe:"+ str(probe_count) + "/" + str(len(probe_faces)))
           

rr_scores = []

eval_count = 0
for probe in probes:
    positives = []
    negatives = []
    eval_count+=1

    print("comparision:"+ str(eval_count) + "/" + str(len(probes)))
    probe_id = df_val[df_val["FACE_ID"]== probe].SUBJECT_ID.values[0]

    for model in models:
        dist = (-DISTANCE_FUNCTION(models[model], probes[probe]))

        if model == probe_id:
            positives.append(dist)
        else:
            negatives.append(dist)
            
    if positives:
        positive_s = numpy.array(positives)
        negative_s = numpy.array(negatives)

        rr_scores.append((negative_s, positive_s))
        rr = bob.measure.recognition_rate(rr_scores, rank=1)
        
    else:
        print("There is no positive for probe: ", probe)

print("Evaluating")
rr = bob.measure.recognition_rate(rr_scores, rank=1)
print("Recognition Rate: rr",rr)

#plot
bob.measure.plot.cmc(rr_scores)
pyplot.title("CMC curve")
pyplot.savefig("/home/user/anushri/LRFR/Face_Rec_exp/cmc_Curve_OG.png")
pyplot.show()

#pickling 
#with open("/home/user/anushri/LRFR/Face_Rec_exp/scores_LR.txt", "wb") as fp:   #Pickling
#   pickle.dump(rr_scores, fp)
 



