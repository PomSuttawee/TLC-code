import numpy as np

def _calculate_average_intensity(image: np.ndarray) -> np.ndarray:
    """
    Compute the average (inverted) intensity per column of a TLC image.
    The inversion is 255 - actual intensity to highlight dark pixels as high intensity.
    """
    sum_intensity = np.sum(image, axis=0)
    count_color_pixel = np.sum(np.where(image > 0, 1, 0), axis=0)
    safe_count_color_pixel = np.where(count_color_pixel == 0, 1, count_color_pixel)
    average_intensity = (255 - (sum_intensity / safe_count_color_pixel)).astype(int)
    average_intensity[count_color_pixel == 0] = 0
    return average_intensity

def _calculate_minima(intensity: np.ndarray) -> np.ndarray:
    """
    Identify indices where intensity transitions from zero to non-zero and vice versa.
    Returns sorted minima indices.
    """
    threshold_intensity = np.where(intensity > 0, 1, 0)
    zero_to_non_zero = np.where((threshold_intensity[:-1] == 0) & (threshold_intensity[1:] != 0))[0]
    non_zero_to_zero = np.where((threshold_intensity[:-1] != 0) & (threshold_intensity[1:] == 0))[0] + 1
    minima_index = np.sort(np.concatenate((zero_to_non_zero, non_zero_to_zero)))
    return minima_index

def _calculate_peak_area(intensity: np.ndarray, minima: np.ndarray) -> np.ndarray:
    """
    Calculate the integrated peak area for each pair of minima indices.
    Assumes minima has an even number of elements.
    """
    peak_area = []
    for index_minima in range(0, len(minima)-1, 2):
        peak_area.append(np.trapz(intensity[minima[index_minima]: minima[index_minima + 1] + 1]))
    return np.array(peak_area)

def calculate_peak_area_from_image(image: np.ndarray) -> np.ndarray:
    """
    Calculate the peak area of the image.
    """
    return _calculate_peak_area(_calculate_average_intensity(image), _calculate_minima(_calculate_average_intensity(image)))