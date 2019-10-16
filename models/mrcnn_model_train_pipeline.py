import mrcnn.model as modellib
from data_processing import set_seed_everywhere
from data_processing import get_fold, get_kfold_cv_splits, preprocess_img_dataframe, read_json_conf_file
from data_processing import DatasetProcessor
from models import MrcnnConfig
import os
import warnings
warnings.filterwarnings("ignore")

# Retrieving main model parameters from config file
conf_file = read_json_conf_file('mrcnn_config.json')
paths, learning_params, model_params = conf_file['paths'], conf_file['learning_params'], conf_file['model_params']

# Setting seed in modules for results reproducibility
set_seed_everywhere(learning_params['SEED'])

# Preparing train/validation datasets
image_df = preprocess_img_dataframe(paths['PATH_TO_IMG_DF'])
splits = get_kfold_cv_splits(image_df)
train_df, valid_df = get_fold(image_df, splits, learning_params['CURRENT_FOLD'])

if learning_params['NORMAL_CASE_TRAINING']:
    train_dataset = DatasetProcessor(train_df)
    train_dataset.prepare()

    valid_dataset = DatasetProcessor(valid_df)
    valid_dataset.prepare()
else:
    train_data = train_df.sample(10, random_state=learning_params['SEED'])
    print('WARNING!!! Test code in progress. Num of examples for train:', len(train_data))
    if learning_params['SAVE_TRAIN_IMAGES']:  # To retrieve images model trained on
        train_data.to_csv(os.path.join('..', 'data', 'trained_on.csv'))
    train_dataset = DatasetProcessor(train_data)
    train_dataset.prepare()

    valid_dataset = DatasetProcessor(train_data)
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
model.train(train_dataset, valid_dataset,
            learning_rate=learning_params['LR_ALL'],
            epochs=learning_params['EPOCHS_ALL'],
            layers='all',
            augmentation=None)
