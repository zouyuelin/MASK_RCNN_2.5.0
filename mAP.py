
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

import os
import re
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import config
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

dataset_root_path = "train_dataset/"
img_floder = dataset_root_path + "imgs/"
mask_floder = dataset_root_path + "mask/"
mask_infofloder = dataset_root_path + "mask_info/"
imglist = os.listdir(img_floder)
split_rate = 0.8
count = len(imglist)
np.random.seed(10101)
np.random.shuffle(imglist)
train_imglist = imglist[:int(count*split_rate)]
val_imglist = imglist[int(count*split_rate):]

class_names = ['BG','grasper','grasper2','grasper3','irrigator','hook','clipper']

image_path = "./train_dataset/imgs/2424.jpg"


class InferenceConfig(config.Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    NAME = "Inference"
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = len(class_names)  # background + your shapes

    #DETECTION_MIN_CONFIDENCE = 0

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 60

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    LEARNING_RATE = 0.002
    
    ALL_EPOCHS = 20

    #set the epochs 
    HEAD_EPOCHS = 3
    ALL_EPOCHS = 20

class ShapesDataset(utils.Dataset):
    """
        override the load_shapes and the load_mask
    """

    def load_shapes(self, count, img_floder, mask_floder, imglist, creatnpzfile:bool=True):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        # 紫色5，红色1，绿色2，黄色3，蓝色4,蓝青6,
        
        self.add_class("shapes", 1, "grasper")
        self.add_class("shapes", 2, "grasper2")
        self.add_class("shapes", 3, "grasper3")
        self.add_class("shapes", 4, "irrigator")
        self.add_class("shapes", 5, "hook")
        self.add_class("shapes", 6, "clipper")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = os.path.join(img_floder,img)
                mask_path = os.path.join(mask_floder,img_name+".png")
                #save the mask infomation with numpy
                mask_info = None
                
                if not os.path.exists(os.path.join(mask_infofloder,"{}.npz".format(img_name))):
                    mask_info = self.load_mask_pre(i,mask_path)
                    np.savez(os.path.join(mask_infofloder,img_name),mask_ = mask_info[0], id_=mask_info[1])
                else:
                    data = np.load(os.path.join(mask_infofloder,"{}.npz".format(img_name)))
                    mask_info = data['mask_'],data['id_']

                self.add_image("shapes", image_id=i, path=img_path, name=img_name, mask_path=mask_path, mask_info=mask_info)
                sys.stdout.write('-------creating the np file:--%s-------------pross:--%.4f%%--'%(os.path.join(mask_infofloder,"{}.npz".format(img_name)),
                                                                                            (i+1)/float(count)*100))
                sys.stdout.write('\r')
                sys.stdout.flush()

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]['mask_info']
        mask_, id_ = info

        return mask_, id_ #mask.astype(np.bool)
    
    def load_mask_pre(self, image_id, mask_path):
        """Generate instance masks for shapes of the given image ID.
        """
        img = Image.open(mask_path)
        colors = img.getcolors()
        n_dim = np.shape(colors)
        num_obj = n_dim[0]-1 #not include the background

        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, colors)

        # Map class names to class IDs.
        class_ids = []
        for i in range(num_obj):
            class_ids.append(colors[i+1][1])

        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32) #mask.astype(np.bool)

    def draw_mask(self, num_obj, mask, image, colors):

        #single thread
        #time_start = time.time()
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == colors[index+1][1]:
                        mask[j, i, index] =1
        #time_end = time.time()
        #time_c= time_end - time_start   #运行所花时间
        #print('time cost at 1', time_c, 's')

        return mask

myconfig = InferenceConfig()
myconfig.display()

dataset_val = ShapesDataset()
dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist)
dataset_val.prepare()

# ## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=myconfig)


# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 50)
APs = []
Precisions = []
Recalls = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, myconfig,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, myconfig), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =  utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    Precisions.append(precisions)
    Recalls.append(recalls)

print("mAP: ", np.mean(APs))

