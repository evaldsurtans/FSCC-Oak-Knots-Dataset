import csv
import json
import os
import pdb
import pickle
import cv2
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import random
import time
import math
import numpy as np
from loguru import logger

from modules.file_utils import FileUtils


class DataUtils(object):

    @staticmethod
    def get_bboxes_from_image(np_x):
        bboxes = []
        np_x = np.array(np_x)
        np_x[np_x >= 0.5] = 255
        np_x[np_x < 0.5] = 0
        cv_label = np_x.astype(np.uint8)
        contours = cv2.findContours(cv_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append([x, y, x + w, y + h])
        return bboxes

    @staticmethod
    def is_bool_true(value):
        return str(value).lower().strip() == 'true'

    @staticmethod
    def save_pickle(obj, path):
        try:
            with open(path, "wb" ) as fp:
                FileUtils.lock_file(fp)
                pickle.dump(obj, fp)
                FileUtils.unlock_file(fp)
        except Exception as e:
            logger.exception(e)

    @staticmethod
    def load_pickle(path):
        obj = None
        try:
            with open(path, "rb" ) as fp:
                FileUtils.lock_file(fp)
                obj = pickle.load(fp)
                FileUtils.unlock_file(fp)
        except Exception as e:
            logger.exception(e)
        return obj

    @staticmethod
    def save_memmap(np_data, path):
        try:
            if np_data.shape[0] > 0:
                mem = np.memmap(path, mode='w+', dtype=np.float16, shape=np_data.shape)
                mem[:] = np_data[:]
                mem.flush()
                del mem

                struct = {
                    'shape': np_data.shape
                }

                path = path[:len(path) - len('.bin')] + '.json'
                with open(path, 'w') as outfile:
                    json.dump(struct, outfile, indent=4)
        except Exception as e:
            logger.exception(e)

    @staticmethod
    def preprocess_batch(batch):
        input = torch.autograd.Variable(batch['x'].type(torch.FloatTensor))
        output_y = torch.autograd.Variable(batch['y'].type(torch.FloatTensor))
        return input, output_y


    @staticmethod
    def write_to_csv(data_row_dict, path):
        is_existing = os.path.exists(path)
        with open(path, 'w' if not is_existing else 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, data_row_dict.keys())
            if not is_existing:
                dict_writer.writeheader()
            dict_writer.writerow(data_row_dict)