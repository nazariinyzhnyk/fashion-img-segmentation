import mrcnn.model as modellib
from data_processing import set_seed_everywhere
from data_processing import get_fold, get_kfold_cv_splits, preprocess_img_dataframe, read_json_conf_file
from data_processing import DatasetProcessor
from models import MrcnnConfig
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
# Retrieving main model parameters from config file
conf_file = read_json_conf_file(os.path.join('..', 'mrcnn_config.json'))
paths, learning_params, model_params = conf_file['paths'], conf_file['learning_params'], conf_file['model_params']

# Setting seed in modules for results reproducibility
set_seed_everywhere(learning_params['SEED'])


# Preparing train/validation datasets
if learning_params['NORMAL_CASE_TRAINING']:
    image_df = preprocess_img_dataframe(paths['PATH_TO_IMG_DF'])
    splits = get_kfold_cv_splits(image_df)
    train_df, valid_df = get_fold(image_df, splits, learning_params['CURRENT_FOLD'])
else:  # overfit on 10 images
    if learning_params['USE_SAVED_IMAGES']:
        image_df = preprocess_img_dataframe(paths['SAVED_DF_PATH'])
    else:
        image_df = pd.read_csv(paths['PATH_TO_IMG_DF'])
        # Be careful! Seed was setted above. Re-set it for true rand sampling.
        image_df = image_df[image_df.ImageId.isin(list(image_df.ImageId.sample(10)))]
        if not learning_params['USE_SAVED_IMAGES']:
            image_df.to_csv(paths['SAVED_DF_PATH'], index=False)
        image_df = preprocess_img_dataframe(paths['SAVED_DF_PATH'])
    train_df, valid_df = image_df, image_df


train_dataset = DatasetProcessor(train_df)
train_dataset.prepare()

valid_dataset = DatasetProcessor(valid_df)
valid_dataset.prepare()


# Building model from config
config = MrcnnConfig()
config.display()
model = modellib.MaskRCNN(mode='training', config=config, model_dir=paths['MODEL_DIR'])

# Loading pretrained weights
if learning_params['CONTINUE_TRAINING']:
    weights_path = os.path.join(paths["FINAL_WEIGHTS_PATH"], str('mask_rcnn_' + model_params["NAME"] + '.h5'))
    if not os.path.isfile(weights_path):
        weights_path = paths['COCO_WEIGHTS_PATH']
else:
    weights_path = paths['COCO_WEIGHTS_PATH']

print('Using pretrained weights from:', weights_path)

model.load_weights(weights_path, by_name=True,
                   exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# Training only head layers
model.train(train_dataset, valid_dataset,
            learning_rate=learning_params['LR_HEADS'],
            epochs=learning_params['EPOCHS_HEADS'],
            layers='heads',
            augmentation=None)

# Training all layers
if learning_params['TRAIN_ALL_LAYERS']:
    model.train(train_dataset, valid_dataset,
                learning_rate=learning_params['LR_ALL'],
                epochs=learning_params['EPOCHS_ALL'],
                layers='all',
                augmentation=None)
