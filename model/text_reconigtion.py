import cv2
import numpy as np

def is_blue(image):
  hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  low_blue = np.array([125, 0, 0])
  high_blue = np.array([160, 360, 360])
  blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
  return (np.sum(blue_mask > 100) > 4)