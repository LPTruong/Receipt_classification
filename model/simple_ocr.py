#
# You can modify this files
#
import cv2
from preprocess import preprocessing
from logo_recognition import crop_logo
from logo_recognition import image_align
from text_segmentation import remove_non_text, trim_boxes, findBoundingBox, generate_feature_vector, preprocess_text
from variable import _dir, segmenation_blur_params ,segmenation_erode_params
from model import loaded_model
from text_reconigtion import is_blue
import numpy as np
import random

logo_size = (150, 150)
logo_phuclong1 = cv2.imread(_dir + "phuclong.jpg")
logo_phuclong1 = cv2.resize(logo_phuclong1, logo_size)
logo_phuclong2 = cv2.imread(_dir + "phuclong2.JPG")
logo_phuclong2 = cv2.resize(logo_phuclong2, logo_size)
logo_starbuck = cv2.imread(_dir + "starbuck.jpg")
logo_starbuck = cv2.resize(logo_starbuck, logo_size)



class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']

    # TODO: implement find label
    def find_label(self, img):
        cropped = cv2.resize(preprocessing(img), (600, 600), cv2.INTER_NEAREST)
        # cv2.imshow(cropped)
        pred = 4

        for box in crop_logo(cropped.copy()):
            if (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy(), \
                            logo_phuclong2, ratio_thresh=0.65, accept_thresh=3)):
                pred = 3
                break
            elif (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy(), \
                              logo_phuclong1, ratio_thresh=0.7, accept_thresh=3)):
                pred = 3
                break
            elif (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy(), \
                              logo_starbuck, ratio_thresh=0.7, accept_thresh=4)):
                pred = 2
                break
            else:
                pred = 4

        if (pred == 4):
            boxes = remove_non_text(trim_boxes(findBoundingBox(cropped.copy(), \
                                                               segmenation_blur_params, segmenation_erode_params)),
                                    cropped)
            for box in boxes:
                if (not is_blue(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy())):
                    continue
                feature = generate_feature_vector(
                    preprocess_text(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy()))
                if (loaded_model.predict([feature])[0] > 0.5):
                    pred = 0
                    crop_img = cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy()

        return self.labels[random.randint(0, 3)]
