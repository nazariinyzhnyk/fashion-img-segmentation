import itertools
import os
import pandas as pd
import mrcnn.model as modellib
from data_processing import read_json_conf_file, resize_image
from models import MrcnnConfigInference

VISUALIZE = True

conf_file = read_json_conf_file(os.path.join('..', 'mrcnn_config.json'))
paths, learning_params, model_params = conf_file['paths'], conf_file['learning_params'], conf_file['model_params']

inference_config = MrcnnConfigInference()
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=paths['MODEL_DIR'])

weights_path = os.path.join(paths["FINAL_WEIGHTS_PATH"], str('mask_rcnn_' + model_params["NAME"] + '.h5'))
if not os.path.isfile(weights_path):
    weights_path = paths['COCO_WEIGHTS_PATH']

print('Using pretrained weights from:', weights_path)

model.load_weights(weights_path, by_name=True)

sample_df = pd.read_csv(paths['SAVED_DF_PATH'])


def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle


sub_list = []
for i, row in sample_df.iterrows():
    img = resize_image(os.path.join('..', 'data', 'images', row['ImageId']), 512)
    result = model.detect([img])[0]
    if result['masks'].size > 0:
        masks = result['masks']
        for m in range(masks.shape[-1]):
            mask = masks[:, :, m].ravel(order='F')
            rle = to_rle(mask)
            label = result['class_ids'][m] - 1
            sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])
    else:
        print('no segments found')
        sub_list.append([row['ImageId'], '1 1', 23])

sub_list_df = pd.DataFrame(sub_list, columns=['ImageId', 'EncodedPixels', 'ClassId'])
sub_list_df.to_csv('../data/overfitted_preds.csv', index=False)
