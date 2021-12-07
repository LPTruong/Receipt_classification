import cv2
import numpy as np
import pandas as pd
from preprocess import preprocessing
from variable import _dir

logo_size = (150, 150)
title_size = (250, 75)

def image_align(image, logo, ratio_thresh=0.70, accept_thresh=10, is_full=False):
    detector = cv2.AKAZE_create()
    logo_gray = logo
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    if len(logo.shape) == 3:
        logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

    keypoints_logo, descriptors_logo = detector.detectAndCompute(logo_gray, None)
    cropped = image.copy()
    cropped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if is_full == False:
        cropped = cv2.resize(cropped, (logo_gray.shape[1], logo_gray.shape[0]), cv2.INTER_CUBIC)
    cropped = cv2.bilateralFilter(cropped, 5, 20, 20)
    cropped = cv2.filter2D(cropped, -1, sharpen_filter)

    keypoints, descriptors = detector.detectAndCompute(cropped, None)

    if len(keypoints) <= 0:
        return False
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    knn_matches = matcher.knnMatch(np.asarray(descriptors, np.float32), np.asarray(descriptors_logo, np.float32), 2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    if len(good_matches) >= accept_thresh:
        # cv2.imshow("cropped", cropped)
        return True

    # print(len(good_matches))
    return False

def crop_logo(image):
    result_rect = []
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    for threshold in range(0, 250, 20):
        ret, thresh = cv2.threshold(gray.copy(), threshold, 255, cv2.THRESH_BINARY_INV)
        # cv2_imshow(thresh)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            # if the size of the contour is greater than a threshold
            if cv2.contourArea(cnt) > 800 and cv2.contourArea(cnt) < 600 * 600 / 6:
                box = cv2.boundingRect(contours[i])
                if box[2] * box[3] < 600 * 600 / 6:
                    result_rect.append(box)
    return result_rect
