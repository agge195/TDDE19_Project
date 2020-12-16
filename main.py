import os
import sys
import numpy as np
import time
import tensorflow as tf
from dataset_cig_config import CigButtsConfig, InferenceConfig
from dataset_class import CocoLikeDataset
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.utils import compute_ap, compute_matches
import skimage

ROOT_DIR = '../project_files'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)

## Fix tf-gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Find root directory
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def calculate_mAP_F1(dataset, model, cfg, image_paths):
    APs = []
    F1_scores = []
    rs = []

    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id)
        scaled_image = modellib.mold_image(image, cfg)
        sample = np.expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        F1_scores.append((2 * (np.mean(precisions) * np.mean(recalls))) / (np.mean(precisions) + np.mean(recalls)))
        APs.append(AP)

    mAP = np.mean(APs)
    print(mAP)
    #print("Mean average precision: {}").format(mAP)

    return mAP, F1_scores


def tune(dataset_train, dataset_val, config):
    learning_rates = [0.0001, 0.001]
    backbones = ['resnet101', 'resnet50']

    mAPs = []
    stats = []

    print("Starting tuning process")
    all_scores = []
    for bk in backbones:

        config.BACKBONE = bk

        for lr in learning_rates:
            config.LEARNING_RATE = lr
            mAP = train(config, dataset_train, dataset_val, predict_= True)
            info = (bk, lr)

            stats.append(info)
            mAPs.append(mAP)

    max_ = np.amax(mAPs)
    idx = mAPs.index(max_)
    print("Best mAP : ", max_)
    print("best config: ", stats[idx])
    print("done")

def predict(dataset_val):
    inference_config = InferenceConfig()

    # prediction
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    real_test_dir = 'datasets/cig_butts/real_test/'
    image_paths = []
    for filename in os.listdir(real_test_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(os.path.join(real_test_dir, filename))

    scores = []
    for image_path in image_paths:
        img = skimage.io.imread(image_path)
        img_arr = np.array(img)
        results = model.detect([img_arr], verbose=1)
        r = results[0]

        scores.append(r['scores'])

        #visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
        #                            dataset_val.class_names, r['scores'], figsize=(5, 5))


    mAP, F1_scores = calculate_mAP_F1(dataset_val, model, inference_config, image_paths)

    print("mAP: {}".format(mAP))

    return mAP


def train(config, dataset_train, dataset_val, predict_ = False):
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    start_train = time.time()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=4,
                layers='heads')
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')

    if predict_:
        mAP = predict(dataset_val)
        return mAP

def setup_training(tune_ = False, predict_ = False):

    dataset_train = CocoLikeDataset()
    dataset_train.load_data('datasets/cig_butts/train/coco_annotations.json', 'datasets/cig_butts/train/images')
    dataset_train.prepare()

    dataset_val = CocoLikeDataset()
    dataset_val.load_data('datasets/cig_butts/val/coco_annotations.json', 'datasets/cig_butts/val/images')
    dataset_val.prepare()

    config = CigButtsConfig()
    if tune_:
        tune(dataset_train, dataset_val, config)
    else:
        config.display()
        train(config, dataset_train, dataset_val, predict_)

        print("done")





setup_training(predict_= True, tune_= True)

# Confidence when classifying from real_test
"""
resnet101, lr=0.0001, mAP: 0.9
resnet101, lr=0.001, mAP: 0.93
resnet50, lr=0.0001, mAP: 0.17213982283196677
resnet50, lr=0.001, mAP: 0.0 #bug
"""
# TODO 1: Do this for custom dataset -> coco with selected classes?
# TODO 2: Hyperparameter tuning
