import mrcnn.model as modellib
from data_processing import set_seed_everywhere
from data_processing import get_fold, get_kfold_cv_splits, preprocess_img_dataframe
from data_processing import DatasetProcessor
from models import MrcnnConfig
import os
import warnings
warnings.filterwarnings("ignore")


# TODO: pass these arguments to .config file when implemented
COCO_WEIGHTS_PATH = os.path.join('..', 'data', 'weights', 'mask_rcnn_coco.h5')
PATH_TO_IMG_DF = os.path.join('..', 'data', 'train.csv')
MODEL_DIR = os.path.join('..', 'model_dir')
LR_HEADS = 1e-4  # lr decay should be written
LR_ALL = 1e-5
EPOCHS_HEADS = 1  # should be changed to more for real task training
EPOCHS_ALL = 1
CURRENT_FOLD = 0
NORMAL_CASE_TRAINING = False  # fit whole dataset or test on 10 images
SEED = 42


# for reproducibility
set_seed_everywhere(SEED)

image_df = preprocess_img_dataframe(PATH_TO_IMG_DF)
splits = get_kfold_cv_splits(image_df)
train_df, valid_df = get_fold(image_df, splits, CURRENT_FOLD)

if NORMAL_CASE_TRAINING:
    train_dataset = DatasetProcessor(train_df)
    train_dataset.prepare()

    valid_dataset = DatasetProcessor(valid_df)
    valid_dataset.prepare()
else:
    train_data = train_df.sample(10, random_state=SEED)
    print('WARNING!!! Test code in progress. Num of examples for train:', len(train_data))
    train_dataset = DatasetProcessor(train_data)
    train_dataset.prepare()

    valid_dataset = DatasetProcessor(train_data)
    valid_dataset.prepare()

config = MrcnnConfig()
config.display()
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


model.train(train_dataset, valid_dataset,
            learning_rate=LR_HEADS,
            epochs=EPOCHS_HEADS,
            layers='heads',
            augmentation=None)

model.train(train_dataset, valid_dataset,
            learning_rate=LR_ALL,
            epochs=EPOCHS_ALL,
            layers='all',
            augmentation=None)


