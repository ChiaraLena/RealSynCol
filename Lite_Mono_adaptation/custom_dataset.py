from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


def get_custom_matrix(name_dataset):
    if name_dataset == 'SimCol':
        K = np.array([[0.479, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        return K

    elif name_dataset == 'RealSynCol':
        K = np.array([[5.958787202835083008e-01, 0, 0.5, 0],
                      [0, 5.958787202835083008e-01, 0.5, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K

    elif name_dataset == 'C3VD':
        K = np.array([[0.5022275, 0, 0.502016, 0],
                      [0, 0.502275, 0.498430, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K

    else:
        return None



class CustomDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)

        self.K = get_custom_matrix(self.name_custom_dataset)

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class CustomRAWDataset(CustomDataset):
    def __init__(self, *args, **kwargs):
        super(CustomRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):

        if self.name_custom_dataset == 'SimCol':
            f_str = "FrameBuffer_{:04d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(self.data_path, folder, f_str)
            return image_path

        if self.name_custom_dataset == 'RealSynCol':
            f_str = "Frame_{:04d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(self.data_path, folder, f_str)

            return image_path

        if self.name_custom_dataset == 'C3VD':
            f_str = "{:04d}_color{}".format(frame_index, self.img_ext)
            image_path = os.path.join(self.data_path, folder, f_str)

            return image_path

        else:
            return None



