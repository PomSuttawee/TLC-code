import numpy as np

def calculate_average_intensity(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    if direction == 'horizontal':
        sum_intensity = np.sum(image, axis=1)
        count_color_pixel = np.sum(np.where(image > 0, 1, 0), axis=1)
    elif direction == 'vertical':
        sum_intensity = np.sum(image, axis=0)
        count_color_pixel = np.sum(np.where(image > 0, 1, 0), axis=0)
    safe_count_color_pixel = np.where(count_color_pixel == 0, 1, count_color_pixel)
    average_intensity = (255 - (sum_intensity / safe_count_color_pixel)).astype(int)
    average_intensity[count_color_pixel == 0] = 0
    return average_intensity

def calculate_minima(image: np.ndarray) -> np.ndarray:
    intensity = calculate_average_intensity(image)
    threshold_intensity = np.where(intensity > 0, 1, 0)
    zero_to_non_zero = np.where((threshold_intensity[:-1] == 0) & (threshold_intensity[1:] != 0))[0]
    non_zero_to_zero = np.where((threshold_intensity[:-1] != 0) & (threshold_intensity[1:] == 0))[0] + 1
    minima_index = np.sort(np.concatenate((zero_to_non_zero, non_zero_to_zero)))
    return minima_index

def calculate_peak_area(image: np.ndarray) -> np.ndarray:
    intensity = calculate_average_intensity(image)
    minima = calculate_minima(image)
    peak_area = []
    for index_minima in range(0, len(minima)-1, 2):
        peak_area.append(np.trapz(intensity[minima[index_minima]: minima[index_minima + 1] + 1]))
    return np.array(peak_area)