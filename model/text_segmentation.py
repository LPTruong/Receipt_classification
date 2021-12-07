import cv2
from preprocess import preprocessing
import numpy as np
from variable import _dir, segmenation_blur_params ,segmenation_erode_params
from variable import _dir

# TODO: Text segmentation


# TODO: Find all potential text boxes (box everything)


def findBoundingBox(image, segmenation_blur_params, segmenation_erode_params):
    boxes = []
    cropped = preprocessing(image)
    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # erase red color from image
    b, g, gray = cv2.split(cropped)
    for blur_param in segmenation_blur_params:
        for erode_param in segmenation_erode_params:
            thresh = cv2.medianBlur(gray, blur_param)
            thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, erode_param)
            thresh = cv2.erode(thresh, rectKernel, iterations=1)

            # Find the contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    box = cv2.boundingRect(cnt)
                    x, y, w, h = box
                    if w > 3 * h or h > 3 * w:
                        boxes += [box]
    boxes = list(dict.fromkeys(boxes))
    return boxes

# TODO: Remove non-text boxes
# Delete similar boxes
def trim_boxes(boxes):
  to_be_delete = [False]*len(boxes)
  for i,box1 in enumerate(boxes):
      x,y,w,h = box1
      for j,box2 in enumerate(boxes):
        if (i >= j):
          continue
        x2,y2,w2,h2 = box2
        if ((abs(x-x2) + abs(y-y2)) < 20 and (abs(w - w2) + abs(h-h2)) < 20):
          to_be_delete[j] = True;
  new_boxes = []
  for i in range(0, len(boxes)):
    if (to_be_delete[i] == False):
      new_boxes.append(boxes[i])
  boxes = new_boxes
  return boxes

def remove_non_text(boxes, image):
  to_be_delete = [False]*len(boxes)
  for i,box1 in enumerate(boxes):
    crop_box = image[box1[1] : box1[1] + box1[3],\
                        box1[0] : box1[0] + box1[2]]
    crop_box = cv2.cvtColor(crop_box, cv2.COLOR_BGR2GRAY)
    ret,crop_box = cv2.threshold(crop_box,125,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ratio = np.sum(crop_box > 125)/ (crop_box.shape[0] * crop_box.shape[1])
    if (ratio < 0.15 or ratio > 0.55):
      to_be_delete[i] = True;
  new_boxes = []
  for i in range(0, len(boxes)):
    if (to_be_delete[i] == False):
      new_boxes.append(boxes[i])
  boxes = new_boxes
  return boxes

# Text recognition

text_image_shape = (150, 30)


def preprocess_text(text_image):
    text_image = cv2.resize(text_image, text_image_shape)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    text_image = cv2.filter2D(src=text_image, ddepth=-1, kernel=kernel)
    text_image_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    ret, text_image_gray = cv2.threshold(text_image_gray, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h = text_image_gray.shape[0]
    w = text_image_gray.shape[1]
    to_be_delete = []
    culmulative = np.sum(text_image_gray, axis=0)
    for i, col in enumerate(culmulative):
        if col < 1:
            to_be_delete += [i]
    text_image_gray = np.delete(text_image_gray, to_be_delete, axis=1)
    to_be_delete = []
    culmulative = np.sum(text_image_gray, axis=1)
    for j, row in enumerate(culmulative):
        if row < 1:
            to_be_delete += [j]
    text_image_gray = np.delete(text_image_gray, to_be_delete, axis=0)
    text_image_gray = cv2.resize(text_image_gray, text_image_shape)

    return text_image_gray


# TODO: Zoning : divide image into zone and count number of black pixel
def zoning(image, nrows, ncols):
    """Return 1x (nrows x ncols) vector count the pixel in each zones"""
    h, w = image.shape
    result = []
    h = h // nrows
    w = w // ncols
    for i in range(0, nrows):
        for j in range(0, ncols):
            result += [np.sum(image[i * h:(i + 1) * h - 1, j * w:(j + 1) * w - 1] <= 125)]

    return result


# Gabor Histogram Features
def gabor_feature(image):
    res = []
    for i in range(0, 4):
        kern = cv2.getGaborKernel(ksize=(7, 7), theta=i * np.pi / 4,
                                  sigma=4, lambd=10, gamma=0.5)
        image_gabor = cv2.filter2D(image, -1, kern)
        res += zoning(image_gabor, 2, 2)
    return res


# Distance Profile Features
def distance_profile_feature(image, division_x=15, division_y=3):
    h, w = image.shape
    top_distance = [0] * (w // division_y)
    bottom_distance = [h] * (w // division_y)
    head_distance = [0] * (h // division_x)
    tail_distance = [w] * (h // division_x)
    for i in range(0, h):
        for j in range(0, w):
            if image[i, j] > 125:
                top_distance[j // division_y] = max(top_distance[j // division_y], i)
                bottom_distance[j // division_y] = min(bottom_distance[j // division_y], i)
                head_distance[i // division_x] = max(head_distance[i // division_x], j)
                tail_distance[i // division_x] = min(tail_distance[i // division_x], j)
    return [top_distance, bottom_distance, head_distance, tail_distance]


def generate_feature_vector(image):
    result = zoning(image, 3, 4)
    dpf = distance_profile_feature(image)
    gabor_feature(image)
    for side in dpf:
        result = result + side
    return result

