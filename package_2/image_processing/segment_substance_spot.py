import cv2
import numpy as np

def _convert_RGB_to_HSV(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def segment_hsv_threshold_range(image: np.ndarray) -> np.ndarray:
    hsv = _convert_RGB_to_HSV(image)
    lower_threshold = np.array([0, 40, 50])
    upper_threshold = np.array([360, 255, 255])
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
    result = cv2.bitwise_and(image, image, mask=morph)
    return result