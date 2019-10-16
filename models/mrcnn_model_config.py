from mrcnn.config import Config
from data_processing import get_label_names, read_json_conf_file


conf_file = read_json_conf_file('mrcnn_config.json')
conf_file = conf_file['model_params']


class MrcnnConfig(Config):
    NUM_CLASSES = len(get_label_names()) + 1

    NAME = conf_file['NAME']

    GPU_COUNT = conf_file['GPU_COUNT']
    IMAGES_PER_GPU = conf_file['IMAGES_PER_GPU']

    BACKBONE = conf_file['BACKBONE']

    IMAGE_MIN_DIM = conf_file['IMAGE_MIN_DIM']
    IMAGE_MAX_DIM = conf_file['IMAGE_MAX_DIM']

    IMAGE_RESIZE_MODE = conf_file['IMAGE_RESIZE_MODE']
    RPN_ANCHOR_SCALES = tuple(conf_file['RPN_ANCHOR_SCALES'])
    STEPS_PER_EPOCH = conf_file['STEPS_PER_EPOCH']
    VALIDATION_STEPS = conf_file['VALIDATION_STEPS']
