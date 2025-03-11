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
        rf_list.append((image.shape[0] - index) / image.shape[0])
    sorted_rf_list = sorted(rf_list, reverse=True)
    return sorted_rf_list

def visualize_process(image: np.ndarray):
    peak_indices = _find_peak_indices(image)
    rf_dict = {}
    for index in peak_indices:
        rf_dict[index] = (image.shape[0] - index) / image.shape[0]
    
    import cv2
    import matplotlib.pyplot as plt
    for peak_index, rf_value in rf_dict.items():
        cv2.putText(image, str(round(rf_value, 2)), (5, peak_index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.show()