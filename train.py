from datasets import *
import mrcnn.model as modellib
from mrcnn import visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

if __name__ == "__main__":
    # datasetTrain = ShapesDataset(dataPath="./dataset/test/type",datatype="type")
    # datasetTrain.load_shapes()
    # datasetTrain.prepare()

    # exit()
    # MODEL_PATH = "logs/shapes20220228T0235/mask_rcnn_shapes_0500.h5"
    MODEL_PATH = None
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    datasetTrain = ShapesDataset(dataPath="./dataset/")
    datasetTrain.load_shapes()
    datasetTrain.prepare()

    datasetTest = ShapesDataset(dataPath="./dataset/")
    datasetTest.load_shapes()
    datasetTest.prepare()

    config = ShapesConfig(len(datasetTrain.class_info),training=True)
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 2
    config.build()
    config.display()   

    # Load and display random samples
    image_ids = np.random.choice(datasetTrain.image_ids, 4)
    for image_id in image_ids:
        image = datasetTrain.load_image(image_id)
        mask, class_ids = datasetTrain.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, datasetTrain.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)

    if MODEL_PATH != None:
        # model.load_weights(model.find_last(), by_name=True)
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
    model.train(datasetTrain, datasetTest,
                learning_rate=config.LEARNING_RATE,
                epochs=config.HEAD_EPOCHS,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(datasetTrain, datasetTest,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=config.ALL_EPOCHS,
                layers="all")