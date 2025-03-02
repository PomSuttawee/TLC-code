from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from package.image_processing.segmentation import util

def _calculate_vertical_intensity(image: np.ndarray) -> np.ndarray:
    """
    Calculate the average intensity of the image vertically.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Average intensity of the image vertically.
    """
    return np.mean(image, axis=1)

def _remove_contour(list_contour: list, min_area: int) -> list:
    return [contour for contour in list_contour if cv2.contourArea(contour) > min_area]

def _get_contour(image: np.ndarray, min_area: int) -> list:
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_NONE
    list_contour, _ = cv2.findContours(image, mode, method)
    if min_area < 0:
        return list_contour
    return _remove_contour(list_contour, min_area)

def _get_bounding_box(list_contour: list) -> list:
        list_box = [(x, y, w, h) for contour in list_contour for x, y, w, h in [cv2.boundingRect(contour)]]
        return sorted(util.groupBoundingBox(list_box), key=lambda x: x[0])

def _detect_centroids(image: np.ndarray) -> List[int]:
    """
    Detect the centroids of the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        List[int]: Centroids of the image in ascending order.
    """
    binary_image = np.where(image > 0, 255, 0).astype(np.uint8)
    contours = _get_contour(binary_image, 200)
    bounding_box_list = _get_bounding_box(contours)
    return sorted([y + h // 2 for x, y, w, h in bounding_box_list])

def _find_peaks_in_intensity(average_intensity: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Find peaks in the average intensity using the scipy library.

    Args:
        average_intensity (np.ndarray): Average intensity of the image.

    Returns:
        Tuple[np.ndarray, dict]: Peaks in the average intensity and their properties.
    """
    average_intensity = average_intensity.flatten()
    return find_peaks(average_intensity, prominence=10, distance=20)

def calculate_rf_detect_peak(image: np.ndarray) -> List[float]:
    """
    Calculate the RF value of the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        List[float]: RF value of the image.
    """
    vertical_intensity = _calculate_vertical_intensity(image)
    peaks = _find_peaks_in_intensity(vertical_intensity)[0]
    return [round((image.shape[0] - peak) / image.shape[0], 2) for peak in peaks]

def calculate_rf_detect_centroid(image: np.ndarray) -> List[float]:
    """
    Calculate the RF value of the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        List[float]: RF value of the image.
    """
    centroids = _detect_centroids(image)
    return [round((image.shape[0] - centroid) / image.shape[0], 2) for centroid in centroids]

def get_image_with_centroids(image: np.ndarray) -> np.ndarray:
    """
    Get the image with centroids.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image with centroids.
    """
    centroids = _detect_centroids(image)
    rf = calculate_rf_detect_centroid(image)
    image_with_centroid = image.copy()
    for rf_value, centroid in zip(rf, centroids):
        cv2.circle(img = image_with_centroid,
                   center = (image.shape[1] // 2, centroid), 
                   radius = 4, 
                   color = (255),
                   thickness = -1)
        cv2.putText(img = image_with_centroid,
                    text = str(rf_value),
                    org = (image.shape[1] // 2 - 20, centroid - 8),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (255),
                    thickness = 2)
    return image_with_centroid