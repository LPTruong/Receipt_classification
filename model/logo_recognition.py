import cv2
import numpy as np
import pandas as pd
from preprocess import preprocessing
from variable import __dir

logo_size = (150, 150)
title_size = (250, 75)

# TODO: Align two image
pd.set_option('display.max_colwidth', None)


logo_img = cv2.imread(__dir + "starbuck.jpg")
logo_img = cv2.resize(logo_img, logo_size)
logo_gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)

detector = cv2.AKAZE_create()
w_cropped = cv2.imread(__dir + "logo_cropped.png")
w_cropped = cv2.resize(w_cropped, logo_size)
# TODO: Apply filter to sharpen
sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
w_cropped = cv2.filter2D(w_cropped, -1, sharpen_filter)

cropped_bw = cv2.cvtColor(w_cropped, cv2.COLOR_BGR2GRAY)
keypoints_logo, descriptors_logo = detector.detectAndCompute(logo_gray, None)
keypoints, descriptors = detector.detectAndCompute(cropped_bw, None)
keypoint_img = w_cropped

# TODO: Matching
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
knn_matches = matcher.knnMatch(np.asarray(descriptors, np.float32), np.asarray(descriptors_logo, np.float32), 2)

# TODO: Filter matches using the Lowe's ratio test
ratio_thresh = 0.80
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# TODO: Draw matches
img_matches = np.empty((max(cropped_bw.shape[0], logo_gray.shape[0]), cropped_bw.shape[1] + logo_gray.shape[1], 3),
                       dtype=np.uint8)
cv2.drawMatches(cropped_bw, keypoints, logo_gray, keypoints_logo, good_matches, img_matches,
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# TODO: Show detected matches
cv2.imshow("Matches", img_matches)
cv2.waitKey(3000)
print(len(good_matches))


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


file_name = __dir + "highland2.jpg"
image = cv2.imread(file_name, cv2.IMREAD_COLOR)
cropped = cv2.resize(preprocessing(image), (600, 600), cv2.INTER_NEAREST)
cropped = cv2.blur(cropped, (3, 3))


def crop_logo(image):
    result_rect = []
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    for threshold in range(0, 250, 20):
        ret, thresh = cv2.threshold(gray.copy(), threshold, 255, cv2.THRESH_BINARY_INV)
        # cv2_imshow(thresh)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)

        for i, cnt in enumerate(contours):
            # if the size of the contour is greater than a threshold
            if cv2.contourArea(cnt) > 800 and cv2.contourArea(cnt) < 600 * 600 / 6:
                box = cv2.boundingRect(contours[i])
                # cv2_imshow(image[box[1]:box[1]+box[3],box[0]:box[0]+box[2]])
                if box[2] * box[3] < 600 * 600 / 6:
                    result_rect.append(box)
    return result_rect


cropped_cpy = cropped.copy()
for box in crop_logo(cropped):
    cv2.rectangle(cropped_cpy, box, color=456)
cv2.imshow("cropped", cropped_cpy)
cv2.waitKey(0)

# TODO: Match each image with the logo
logo_img = cv2.resize(logo_img, logo_size)
cropped = cv2.resize(preprocessing(image), (600, 600), cv2.INTER_NEAREST)
isLogo = False
for box in crop_logo(cropped):
    if image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]], logo_img):
        isLogo = True
    if isLogo:
        break
print(isLogo)

# Test
# TODO: Test starbuck

logo_img = cv2.imread(__dir + "starbuck.jpg")
logo_img = cv2.resize(logo_img, logo_size)

print('STARBUCKS')

for i in range(1, 14):
    file_name = __dir + "image" + str(i) + ".jpg"
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cropped = cv2.resize(preprocessing(image), (600, 600), cv2.INTER_NEAREST)
    isLogo = False
    for box in crop_logo(cropped):
        if (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]],
                        logo_img, ratio_thresh=0.7, accept_thresh=10)):
            isLogo = True
        if isLogo:
            break
    print("Image ", i, ": ", isLogo)

for i in range(1, 12):
    file_name = __dir + "phuclong" + str(i) + ".jpg"
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cropped = cv2.resize(preprocessing(image), (600, 600), cv2.INTER_NEAREST)
    isLogo = False
    for box in crop_logo(cropped):
        if image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]], logo_img):
            isLogo = True
        if isLogo:
            break
    print("Image ", i, ": ", isLogo)

# TODO: Test Phuc Long

print('PHUC LONG')

logo_img = cv2.imread(__dir + "phuclong.jpg")
logo_img = cv2.resize(logo_img, logo_size)
logo_img2 = cv2.imread(__dir + "phuclong2.JPG")
logo_img2 = cv2.resize(logo_img2, logo_size)
for i in range(1, 13):
    file_name = __dir + "phuclong" + str(i) + ".jpg"
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cropped = cv2.resize(preprocessing(image), (600, 600), cv2.INTER_NEAREST)
    isLogo = False
    for box in crop_logo(cropped):
        if (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]],
                        logo_img2, ratio_thresh=0.70, accept_thresh=4)):
            isLogo = True
        if (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]],
                        logo_img, ratio_thresh=0.65, accept_thresh=3)):
            isLogo = True
        if isLogo:
            break
    print("Image ", i, ": ", isLogo)

logo_img = cv2.imread(__dir + "phuclong.jpg")
logo_img = cv2.resize(logo_img, logo_size)
logo_img2 = cv2.imread(__dir + "phuclong2.JPG")
logo_img2 = cv2.resize(logo_img2, logo_size)
for i in range(1, 14):
    file_name = __dir + "other" + str(i) + ".jpg"
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cropped = cv2.resize(preprocessing(image), (600, 600), cv2.INTER_NEAREST)
    isLogo = False
    for box in crop_logo(cropped):
        if (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]],
                        logo_img2, ratio_thresh=0.70, accept_thresh=4)):
            isLogo = True
        if (image_align(cropped[box[1]:box[1] + box[3], box[0]:box[0] + box[2]],
                        logo_img, ratio_thresh=0.65, accept_thresh=3)):
            isLogo = True
        if isLogo:
            break
    print("Image ", i, ": ", isLogo)

print('PHUC LONG')

