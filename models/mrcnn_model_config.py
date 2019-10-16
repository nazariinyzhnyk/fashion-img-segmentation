from mrcnn.config import Config
from data_processing import get_label_names  # TODO: implement relative path as argument


class MrcnnConfig(Config):  # TODO: implement .config file for automatic model load
    NAME = "fashion"
    NUM_CLASSES = len(get_label_names()) + 1

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = 'resnet101'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 2
