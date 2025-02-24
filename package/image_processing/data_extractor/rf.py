from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def _calculate_vertical_intensity(image: np.ndarray) -> np.ndarray:
    """
    Calculate the average intensity of the image vertically.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Average intensity of the image vertically.
    """
    return np.mean(image, axis=1)

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

def calculate_rf(image: np.ndarray) -> List[float]:
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

def plot_vertical_intensity(image: np.ndarray):
    """
    Plot the average intensity of the image vertically.

    Args:
        image (np.ndarray): Input image.
    """
    vertical_intensity = _calculate_vertical_intensity(image)
    data = vertical_intensity[:, 2]
    peaks, properties = find_peaks(data)
    image_with_peak_lines = image.copy()
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data)
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--')
        image_with_peak_lines = cv2.line(image_with_peak_lines, (0, peak), (image.shape[1], peak), (0, 0, 255), 2)
    plt.vlines(x=peaks, ymin=data[peaks] - properties["prominences"], ymax=data[peaks], color="C1", linestyles="dashed")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color="C1", linestyles="dashed")
    plt.title("Average Vertical Intensity")
    plt.xlabel("Row")
    plt.ylabel("Intensity")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_with_peak_lines, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis('off')
    plt.show()