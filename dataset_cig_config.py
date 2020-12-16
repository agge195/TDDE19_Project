from mrcnn.config import Config

class CigButtsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "cig_butts"

    # Train on 1 GPU and 1 image per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (cig_butt)

    LEARNING_RATE = 0.001
    # Training images dimensions
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # steps per epoch batches
    STEPS_PER_EPOCH = 500

    # validation step batches
    VALIDATION_STEPS = 5

    # The backbone
    BACKBONE = 'resnet101' #test resnet101

    # Some other stuff
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000


class InferenceConfig(CigButtsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    USE_MINI_MASK = False