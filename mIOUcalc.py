import os
import sys
import skimage.io
from mrcnn.mIOU import *
import mrcnn.model as modellib
from datasets import *

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model

def calcMiou(r,data,myconfig:ShapesConfig):   
    mask_gt = np.array(data['mask_'], dtype=np.int)
    id_gt = data['id_']
    id_gt = np.array([int(i) for i in id_gt])
    mask_gt_com = 0
    for i in range(mask_gt.shape[2]):
        mask_gt[:,:,i] = mask_gt[:,:,i]*id_gt[i]
        mask_gt_com = mask_gt_com + mask_gt[:,:,i]

    mask_pre = np.asarray(r['masks'],dtype=np.int)
    id_pre = np.array(r['class_ids'])
    score_pre = r['scores']
    max_index = np.argsort(-score_pre)
    mask_pre_com = 0
    dim = mask_pre.shape[2]
    pre_temp = np.zeros(shape=(mask_pre.shape[0],mask_pre.shape[1]),dtype=np.int)

    for i in range(dim):
        if i == 0 :
            pre_temp = pre_temp + mask_pre[:, :, max_index[i]] * id_pre[max_index[i]]
            continue
        for rows in range(mask_pre.shape[0]):
            for cols in range(mask_pre.shape[1]):
                if pre_temp[rows,cols] == 0 :
                    pre_temp[rows,cols] = mask_pre[rows,cols, max_index[i]] * id_pre[max_index[i]]
    mask_pre_com = pre_temp

    print("id_pre:", id_pre)
    print("id_gt:", id_gt)
    print("r['scores']:", r['scores'])
    temp = IOUMetric(myconfig.NUM_CLASSES)

    iou, miou, ap, dice  = temp.evaluate(mask_pre_com, mask_gt_com)
    return iou,miou,ap, dice

if __name__ == "__main__":
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/shapes20220228T0235/mask_rcnn_shapes_0500.h5")
    dataPath = "dataset/test/binary/"
    Test_num = 50 #all

    datasetTest = ShapesDataset(dataPath=dataPath,datatype=dataPath.split("/")[2])
    datasetTest.load_shapes()
    datasetTest.prepare()  

    myconfig = ShapesConfig(len(datasetTest.class_info),training=False)
    myconfig.display()

    ## Create Model and Load Trained Weights
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=myconfig)

    # Load weights trained on dataset
    model.load_weights(MODEL_DIR, by_name=True)

    mIOU = []
    IOU =[]
    Dice=[]
    index = 0
    for image_name_ in datasetTest.image_info:
        if(index >= Test_num):
            break
        image_name = image_name_["path"]
        image = skimage.io.imread(image_name)
        # Run detection
        results = model.detect([image], verbose=1)
        # Get result
        r = results[0]
        # Load image and ground truth data
        path_ = image_name.split("src")[0]
        src_ = image_name.split("src")[1]

        data = np.load(path_+"mask_Info"+src_.split(".")[0]+".npz", allow_pickle=True)  # allow_pickle=True

        print("the number of classes is:", myconfig.NUM_CLASSES)

        iou,miou,ap, dice = calcMiou(r,data,myconfig)
        mIOU.append(miou)
        IOU.append(iou)
        Dice.append(dice)
        index = index + 1
        
    final_miou = np.mean(mIOU)
    final_dice = np.mean(Dice)
    print("The last iou ({}) is {}".format(dataPath.split("/")[2],final_miou))
    print("Dice is {}".format(final_dice))