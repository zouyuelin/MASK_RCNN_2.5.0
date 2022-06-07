import os
from PIL import Image
import numpy as np
import sys
from mrcnn.config import Config
from mrcnn import utils
import sys

ratio = 32
# testType = {"1":,"2":,"3":,"4":,"6":}

# ## Configurations
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """  
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Batch size is 2 (GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + your shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 60

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    SAVE_EPOCHES_FREQ_PER_STEP = 5

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10

    LEARNING_RATE = 0.0006

    # set the epochs
    HEAD_EPOCHS = 70
    ALL_EPOCHS = 600

    # backbone
    BACKBONE = "resnet101"  # resnet101

    def __init__(self,num_class,training=True):
        self.NUM_CLASSES = num_class
        if training == False:
            self.IMAGES_PER_GPU = 1
        super().__init__()
    def build(self):
        super().__init__()

class ShapesDataset(utils.Dataset):
    """
        override the load_shapes and the load_mask
    """
    def __init__(self, dataPath:str,training=True):
        super().__init__()
        assert dataPath.strip() !='',"please confirm the data Path**"

        self.training = training
        self.dataPath = dataPath
        self.lablePath = os.path.join(dataPath,"label")
        self.srcPath = os.path.join(dataPath,"src")
        self.mask_InfoFloder = os.path.join(dataPath,"mask_Info")
        if not os.path.exists(self.mask_InfoFloder):
            os.mkdir(self.mask_InfoFloder)

        self.imglist = os.listdir(self.srcPath)
        np.random.shuffle(self.imglist)

    def load_shapes(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes -2017 type dataset

        self.add_class("shapes", 1, "grasper")
        self.add_class("shapes", 2, "grasper2")
        self.add_class("shapes", 3, "grasper3")
        self.add_class("shapes", 4, "irrigator")
        self.add_class("shapes", 5, "hook")
        self.add_class("shapes", 6, "clipper")

        for i in range(len(self.imglist)):
            img = self.imglist[i]
            if img.endswith(".jpg") or img.endswith(".png"):
                img_name = img.split(".")[0]
                img_path = os.path.join(self.srcPath, img)
                mask_path = os.path.join(self.lablePath, img_name + ".png")
                # save the mask infomation with numpy
                mask_info = None
                if not os.path.exists(os.path.join(self.mask_InfoFloder, "{}.npz".format(img_name))):
                    mask_info = self.load_mask_pre( mask_path)
                    np.savez(os.path.join(self.mask_InfoFloder, img_name), mask_=mask_info[0], id_=mask_info[1])
                else:
                    data = np.load(os.path.join(self.mask_InfoFloder, "{}.npz".format(img_name)))  # ,allow_pickle=True
                    mask_info = data['mask_'], data['id_']

                self.add_image("shapes", image_id=i, path=img_path, name=img_name, mask_path=mask_path,
                               mask_info=mask_info)

                sys.stdout.write('\r creating the np file:--%s-----pross:--%.4f%%--' % (
                                            os.path.join(self.mask_InfoFloder, 
                                                    "{}.npz".format(img_name)),
                                             (i + 1) / float(len(self.imglist)) * 100))
                sys.stdout.flush()

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]['mask_info']
        mask_, id_ = info
        return mask_, id_ # mask.astype(np.bool)

    def load_mask_pre(self, mask_path):
        """Generate instance masks for shapes of the given image ID.
        """
        img = Image.open(mask_path)
        img = img.convert("L")
        colors = img.getcolors()
        n_dim = np.shape(colors)
        num_obj = n_dim[0] - 1  # not include the background

        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, colors)

        # Map class names to class IDs.
        class_ids = []
        if self.training == True:
            for i in range(num_obj):
                class_ids.append(int(colors[i + 1][1]/ratio))
        else:
            for i in range(num_obj):
                class_ids.append(int(colors[i + 1][1]/ratio))

        return mask.astype(np.bool), class_ids  # mask.astype(np.bool)

    def draw_mask(self, num_obj, mask, image:Image.Image, colors):
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == colors[index + 1][1]:
                        mask[j, i, index] = 1
        return mask
