
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

import os
import sys
import numpy as np
from mrcnn import utils
import mrcnn.model as modellib
from datasets import *
from mIOUcalc import calcMiou
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

def mAP(image_id,myconfig,dataset_val_:ShapesDataset):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val_, myconfig,
                                image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, myconfig), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    # results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =  utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                        r["rois"], r["class_ids"], r["scores"], r['masks'])

    return AP, precisions, recalls

if __name__ == "__main__":
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/shapes20220228T0235/mask_rcnn_shapes_0500.h5")
    dataPath = "dataset/test/binary/"
    Test_num = 50 #all

    dataset_val = ShapesDataset(dataPath=dataPath,datatype=dataPath.split("/")[2])
    dataset_val.load_shapes()
    dataset_val.prepare()

    myconfig = ShapesConfig(len(dataset_val.class_info),training=False)
    myconfig.display()

    # ## Create Model and Load Trained Weights

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=myconfig)

    # Load weights trained on MS-COCO
    model.load_weights(MODEL_DIR, by_name=True)

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = dataset_val.image_ids
    APs = []
    Precisions = []
    Recalls = []
    index = 0
    for image_id in image_ids:
        if(index >= Test_num):
            break
        # Load image and ground truth data
        AP, precisions, recalls = mAP(image_id,myconfig,dataset_val)
        APs.append(AP)
        Precisions.append(np.mean(precisions))
        Recalls.append(np.mean(recalls))
        index = index + 1

    print("precision: {}".format(np.mean(Precisions)))
    print("recalls: {}".format(np.mean(Recalls)))
    print("mAP: ", np.mean(APs))

