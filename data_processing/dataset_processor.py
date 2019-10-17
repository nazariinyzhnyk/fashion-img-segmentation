from mrcnn import utils
import os
import cv2
import numpy as np
from data_processing import resize_image, get_label_names


class DatasetProcessor(utils.Dataset):
    def __init__(self, df):
        super().__init__(self)
        self.label_names = get_label_names(os.path.join('..', 'data', 'label_descriptions.json'))

        # Add classes
        for i, name in enumerate(self.label_names):
            self.add_class("fashion", i + 1, name)

        # Add images
        for i, row in df.iterrows():
            self.add_image("fashion",
                           image_id=row.name,
                           path=os.path.join('..', 'data', 'images', row.name),
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'], width=row['Width'])
        self.img_size = 512

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [self.label_names[int(x)] for x in info['labels']]

    def load_image(self, image_id):
        img_path = self.image_info[image_id]['path']
        return resize_image(img_path, self.img_size)

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros((self.img_size, self.img_size, len(info['annotations'])), dtype=np.uint8)
        labels = []

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height'] * info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            mask[:, :, m] = sub_mask
            labels.append(int(label) + 1)

        return mask, np.array(labels)
