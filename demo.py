
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mrcnn.model as modellib
from mrcnn import visualize
from datasets import *
from mIOUcalc import calcMiou
from mrcnn.mIOU import *


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

if __name__ == "__main__":
    # image path
    image_path = "dataset/test/binary/src/01156.jpg"
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/shapes20220228T0235/mask_rcnn_shapes_0500.h5")  
    path_ = image_path.split("src")[0]
    src_ = image_path.split("src")[1]

    datasetTest = ShapesDataset(dataPath=path_,datatype=image_path.split("/")[2])
    datasetTest.load_shapes()
    datasetTest.prepare()  


    # Mask dataPath
    data = np.load(path_+"mask_Info"+src_.split(".")[0]+".npz")#,allow_pickle=True


    myconfig = ShapesConfig(len(datasetTest.class_info),training=False)
    myconfig.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=myconfig)
    # Load weights trained on MS-COCO
    model.load_weights(MODEL_DIR, by_name=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                datasetTest.class_names, r['scores'])

    # Compute VOC-Style mAP @ IoU=0.5
    iou,miou,ap, dice  = calcMiou(r,data,myconfig)
    print("The iou is {}, the mIou is {}, ap is {}, dice is {}".format(iou,miou, ap, dice))