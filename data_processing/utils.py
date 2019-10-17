import cv2
import os
import json
import random
import numpy
import tensorflow as tf
from sklearn.model_selection import KFold
# from skmultilearn.model_selection import iterative_train_test_split  # TODO: implement multilabel stratification
import pandas as pd


def resize_image(image_path, img_size):
    img = cv2.imread(image_path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def get_label_names(path_to_json_file):
    with open(path_to_json_file) as f:
        return [x['name'] for x in json.load(f)['categories']]


def set_seed_everywhere(seed):
    numpy.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def get_kfold_cv_splits(df, n_split=5):
    kf = KFold(n_splits=n_split, random_state=42, shuffle=True)
    return kf.split(df)


def get_fold(df, splts, fold):
    for i, (train_index, valid_index) in enumerate(splts):
        if i == fold:
            return df.iloc[train_index], df.iloc[valid_index]


def preprocess_img_dataframe(path_to_df):
    segment_df = pd.read_csv(path_to_df)
    segment_df['CategoryId'] = segment_df['ClassId'].apply(split_by_multilabel_cats)  #.str.split('_').str[0]
    df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
    size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
    df = df.join(size_df, on='ImageId')
    return df


def split_by_multilabel_cats(row):
    row = str(row)
    if '_' in row:
        return row.split('_')[0]
    else:
        return row


def read_json_conf_file(path_to_file):
    with open(path_to_file, 'r') as jfile:
        return json.load(jfile)
