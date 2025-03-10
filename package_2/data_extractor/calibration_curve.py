import numpy as np

def _calculate_average_value(image: np.ndarray) -> np.ndarray:
    sum_value = np.sum(image, axis=0)
    count_color_pixel = np.sum(np.where(image > 0, 1, 0), axis=0)
    safe_count_color_pixel = np.where(count_color_pixel == 0, 1, count_color_pixel)
    average_value = (255 - (sum_value / safe_count_color_pixel)).astype(int)
    average_value[count_color_pixel == 0] = 0
    return average_value

def _calculate_minima(intensity: np.ndarray) -> np.ndarray:
    threshold_intensity = np.where(intensity > 0, 1, 0)
    zero_to_non_zero = np.where((threshold_intensity[:-1] == 0) & (threshold_intensity[1:] != 0))[0]
    non_zero_to_zero = np.where((threshold_intensity[:-1] != 0) & (threshold_intensity[1:] == 0))[0] + 1
    minima_index = np.sort(np.concatenate((zero_to_non_zero, non_zero_to_zero)))
    # insert 0 and len(intensity) to minima_index
    minima_index = np.insert(minima_index, 0, 0)
    minima_index = np.append(minima_index, len(intensity)-1)
    return minima_index

def _calculate_peak_area(average_along_y: np.ndarray, minima: list) -> list:
    peak_area = []
    for minima_index in range(0, len(minima)-1, 2):
        peak_area.append(np.trapz(average_along_y[minima[minima_index]:minima[minima_index+1]]))
    return peak_area

def _fit_peak_area_to_concentration(peak_area: list, concentration_list: list) -> list:
    return np.polyfit(concentration_list, peak_area, 1)

def _calculate_r_squared(peak_area: list, concentration_list: list, coef: float, intercept: float) -> float:
    y_hat = coef * np.array(concentration_list) + intercept
    y_bar = np.mean(peak_area)
    ss_tot = np.sum((peak_area - y_bar) ** 2)
    ss_res = np.sum((peak_area - y_hat) ** 2)
    return 1 - ss_res / ss_tot

def calculate_calibration_curve(horizontal_lane_images: list, concentration_list: list):
    best_fit_line_list = []
    for horizontal_lane_image in horizontal_lane_images:
        average_value_along_y = _calculate_average_value(horizontal_lane_image)
        minima = _calculate_minima(average_value_along_y)
        peak_area = _calculate_peak_area(average_value_along_y, minima)
        
        new_concentration_list = concentration_list[-len(peak_area):]
        slope, intercept = _fit_peak_area_to_concentration(peak_area, new_concentration_list)
        r_squared = _calculate_r_squared(peak_area, new_concentration_list, slope, intercept)
        best_fit_line_list.append([slope, intercept, r_squared])
    return best_fit_line_list