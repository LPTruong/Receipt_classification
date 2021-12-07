import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from variable import _dir, image_shape

# 1. Image preprocessing
# TODO: Assumption : receipt have white color, separate from background, rectangle shape

pd.set_option('display.max_colwidth', None)
# 1.1. Edge Detection
# TODO: Use bilateral filter the image to reduce noise,
#  blur and dilated to mostly clear text on the receipt,
#  convert to HSV and keep the S dimension to separate white region,
#  then apply adaptive threshold -> Canny

def background_threshold(test_image, blur_param=7, type_threshold=1):
    """Thresholding to separate the background"""
    blurred = cv2.bilateralFilter(test_image, blur_param, 70, 70)
    blurred = cv2.medianBlur(blurred, blur_param)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    erosion = cv2.erode(blurred, kernel, iterations=1)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 12))
    dilated = cv2.dilate(erosion, rectKernel)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, rectKernel)

    lab = cv2.cvtColor(closing, cv2.COLOR_BGR2HSV)
    lab = cv2.split(lab)[1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = clahe.apply(lab)
    if type_threshold == 1:
        mask = cv2.adaptiveThreshold(lab, 250, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 13)
    else:
        (thresh, mask) = cv2.threshold(lab, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def edge_detection(gray, blur_param=7, aperture_size=3, type_threshold=1):
    """Return edge image after detect"""
    binary = background_threshold(gray, blur_param=blur_param, type_threshold=type_threshold)
    bordered = cv2.copyMakeBorder(binary, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, 0)
    edged = cv2.Canny(bordered, 100, 200, apertureSize=aperture_size)
    return edged

# 1.2. Hough transform
def intersection(line1, line2):
    """
    Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""
    intersections = []
    margin = image_shape[0] / 3
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    if abs(line1[1] - line2[1]) < 1e-3:
                        continue
                    point = intersection(line1, line2)
                    if (-margin <= point[0] <= image_shape[0] + margin
                            and -margin <= point[1] <= image_shape[1] + margin):
                        intersections.append(intersection(line1, line2))
    return intersections


def detect_hough_line(edged):
    lines = cv2.HoughLines(edged, rho=1, theta=np.pi / 180, threshold=image_shape[0] // 6)
    if lines is None:
        lines = np.array([[[0, 0]], [[0, np.pi / 2]], [[600, 0]], [[600, np.pi / 2]]])
        return lines
    if len(lines) < 4:
        lines = np.array([[[0, 0]], [[0, np.pi / 2]], [[600, 0]], [[600, np.pi / 2]]])
    lines = trim_line(lines)
    return lines


def trim_line(lines):
    # Remove nearby lines
    to_remove = []
    for i in range(0, len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            angle_diff = min(abs(line1[1] - line2[1]), abs(line1[1] - line2[1] - np.pi))
            if angle_diff < 0.2:
                if angle_diff < 5e-2:
                    if abs(line1[0] - line2[0]) < 80:
                        to_remove.append(j)
                    continue

                x_in, y_in = intersection(line1, line2)
                if 0 <= x_in <= 600 and 0 <= y_in <= 600:
                    to_remove.append(j)
                    continue
    lines = np.delete(lines, to_remove, axis=0)
    return lines

# 1.3. Select detected lines to crop
# TODO: Cluster hough lines into two sets base on theta value

def separate(lines):
    thetas = [i[0][1] for i in lines]
    thetas = [[min(i, abs(np.pi - i))] for i in thetas]

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(thetas)

    v_set = []
    h_set = []
    for i in range(0, len(lines)):
        if kmeans.labels_[i] == 0:
            v_set += [lines[i]]
        else:
            h_set += [lines[i]]
    return v_set, h_set


# TODO: For each two lines in vertical set and two lines in horizontal set,
#  compute four intersections, and rate them

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) * np.linalg.norm(bc) == 0:
        return float('nan')
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(max(min(cosine_angle, 1), -1))
    return angle


def sharpness(intersections):
    """Return sharpness of a polygon"""
    res = 0
    for i in range(0, 4):
        res += abs(np.pi / 2 - angle(intersections[i], intersections[(i + 1) % 4],
                                     intersections[(i + 2) % 4]))
    if np.isnan(res):
        res = 0
    return res


def distance_to_center(intersections):
    """L2 distance from center of intersections to middle image"""
    center = intersections[0] / 4
    for i in range(1, 4):
        center += intersections[i] / 4
    image_center = np.array([300, 300])
    return np.linalg.norm(center - image_center)


def segments(p):
    return zip(p, p[1:] + [p[0]])


def area_four_point(intersections, n):
    """Return area of a polygon, point given in clockwise order"""
    return 0.5 * abs(sum(x0 * y1 - x1 * y0
                         for ((x0, y0), (x1, y1)) in segments(intersections)))


def score_four_points(intersections):
    rect = order_points(intersections)
    point = 0
    # sharpness of each corner
    sharpnes = sharpness(rect) / (2 * np.pi)
    # the distance to the center of gravity
    distance = distance_to_center(rect) / 282.0
    # ratio of the area with 1/3 image
    area = area_four_point(rect, 4) / (600 * 600)
    for i in range(0, 4):
        if angle(intersections[i], intersections[(i + 1) % 4], intersections[(i + 2) % 4]) < np.pi / 8:
            return 1e9
    if area < 0.33:
        return 1e9
    if area_four_point(rect, 4) < 50:
        return 1e9
    # print(intersections)
    # print(sharpness,' ', distance,' ', area)
    return 100 * sharpnes + 100 * distance + 200 * area


def get_best_corners(lines):
    v_set, h_set = separate(lines)

    best_corners = np.array([[0, 0], [0, 600], [600, 0], [600, 600]])
    best_score = 1e9
    for i, line_i in enumerate(v_set):
        for j, line_j in enumerate(v_set[i + 1:]):
            for p, line_p in enumerate(h_set):
                for q, line_q in enumerate(h_set[p + 1:]):
                    intersections = segmented_intersections([line_i, line_j, line_p, line_q])
                    if len(intersections) < 4:
                        continue;
                    intersections = np.array(intersections[:4])
                    score = score_four_points(intersections)
                    if score < best_score:
                        best_corners = intersections
                        best_score = score
    # print(best_score)
    return best_corners

# Transform receipt to the bird eye view
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


# Highlight text
def text_highligh(image):
    blurred = cv2.bilateralFilter(image, 5, 20, 20)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = clahe.apply(blurred)
    mask = cv2.adaptiveThreshold(blurred, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 51)
    return mask

# Preprocessing function
def preprocessing(__image):
    image_resize = cv2.resize(__image, (600, 600), cv2.INTER_CUBIC)
    # blur, ap, edge
    edge_params = [(5, 3, 0), (7, 3, 1), (7, 3, 0), (11, 5, 1)]
    best_score = 1e9
    best_crop = np.array([[0, 0], [0, 600], [600, 0], [600, 600]])
    for edge_param in edge_params:
        edged = edge_detection(image_resize, blur_param=edge_param[0], aperture_size=edge_param[1],
                               type_threshold=edge_param[2])
        lines = trim_line(detect_hough_line(edged))
        if len(lines) < 2:
            lines = [[0, 0], [0, np.pi / 2], [600, 0], [600, np.pi / 2]]
            continue
        hough_intersections = get_best_corners(lines)
        score = score_four_points(hough_intersections)
        if score < best_score:
            best_crop = hough_intersections
            best_score = score

    cropped = four_point_transform(image_resize, best_crop)
    if cropped.shape[0] < cropped.shape[1] - 100:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cropped = cv2.resize(cropped, (600, 600), cv2.INTER_CUBIC)

    return cropped