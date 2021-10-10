
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

import os
import sys
import random
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from PIL import Image
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#是否接最后一次开始训练
continue_train_from_last = False

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(MODEL_PATH):
    print("there is no {}".format(MODEL_PATH))
else:
    print("find the file: {}".format(MODEL_PATH))

dataset_root_path = "train_dataset/"
img_floder = dataset_root_path + "imgs/"
mask_floder = dataset_root_path + "mask/"
mask_infofloder = dataset_root_path + "mask_info/"
imglist = os.listdir(img_floder)

split_rate = 0.9
count = len(imglist)
np.random.seed(10101)
np.random.shuffle(imglist)
train_imglist = imglist[:int(count*split_rate)]
val_imglist = imglist[int(count*split_rate):]


# ## Configurations

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    #Batch size is 2 (GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + your shapes

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
    
    #set the epochs 
    HEAD_EPOCHS = 3
    ALL_EPOCHS = 30

    #backbone
    #BACKBONE = "resnet50"
    
config = ShapesConfig()
config.display()


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



# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(len(train_imglist), img_floder, mask_floder, train_imglist,)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist)
dataset_val.prepare()

# Load and display random samples

image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    



# ## Create Model

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# if you want train continue, use the function find_last()
#
if continue_train_from_last:
    model.load_weights(model.find_last(), by_name=True)
else:
    model.load_weights(MODEL_PATH, by_name=True,
                      exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])


# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=config.HEAD_EPOCHS, 
            layers='heads')


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=config.ALL_EPOCHS, 
            layers="all")

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

image_ids = np.random.choice(dataset_val.image_ids, 50)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =  utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
