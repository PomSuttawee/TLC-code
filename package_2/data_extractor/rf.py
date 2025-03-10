import numpy as np
from scipy.signal import find_peaks

def _find_peak_indices(image: np.ndarray):
    binarized_image = image.copy()
    binarized_image[binarized_image > 0] = 1
    y_sum_binarized = np.sum(binarized_image, axis=1)
    peak_index = find_peaks(y_sum_binarized)[0]
    return peak_index

def calculate_rf_from_image(image: np.ndarray):
    peak_indices = _find_peak_indices(image)
    rf_list = []
    for index in peak_indices:
        rf_list.append(index / image.shape[0])
    sorted_rf_list = sorted(rf_list, reverse=True)
    return sorted_rf_list