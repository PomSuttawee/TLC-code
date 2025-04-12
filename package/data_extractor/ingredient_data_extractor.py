import numpy as np

def extract_data(image: np.ndarray, bounding_boxes: list, concentration_list: list) -> dict:
    """
    Extract data from the image.
    """
    assert isinstance(image, np.ndarray), "Image must be a numpy array."
    assert isinstance(bounding_boxes, list), "Bounding boxes must be a list."
    assert isinstance(concentration_list, list), "Concentration list must be a list."
    
    rf_data = calculate_rf(image, bounding_boxes)
    calibration_curve = calculate_calibration_curve(image, bounding_boxes, concentration_list)
    data = {}
    for i in range(len(bounding_boxes)):
        data[f"H{i+1}"] = {"rf": rf_data[i], "calibration_curve": calibration_curve[i]}
    return data

def calculate_rf(image: np.ndarray, bounding_boxes: list) -> list:
    """
    Calculate the Rf value of the spots in the image.
    """
    highest_concentration_boxes = _get_right_most_boxes(bounding_boxes)
    total_distance = image.shape[0]
    rf_data = []
    for boxes in highest_concentration_boxes:
        center_y = boxes[1] + boxes[3] // 2
        rf = (total_distance - center_y) / total_distance
        rf_data.append(rf)
    return rf_data

def _get_right_most_boxes(bounding_boxes: list) -> list:
    """
    Get the right most bounding boxes.
    """
    return [boxes[-1] for boxes in bounding_boxes]

def calculate_calibration_curve(image: np.ndarray, bounding_boxes: list, concentration_list: list) -> list:
    """
    Calculate the calibration curve of the spots in the image.
    """
    calibration_curve = []
    for i, horizontal_boxes in enumerate(bounding_boxes):
        peak_areas = _calculate_peak_areas(image, horizontal_boxes)
        new_concentration_list = concentration_list[-len(peak_areas):]
        slope, intercept = np.polyfit(new_concentration_list, peak_areas, 1)
        r_squared = _calculate_r_squared(peak_areas, new_concentration_list, slope, intercept)
        calibration_curve.append((slope, intercept, r_squared))
    return calibration_curve
        
def _calculate_peak_areas(image: np.ndarray, horizontal_boxes: list) -> list:
    """
    Calculate the peak areas of the spots in the image.
    """
    peak_areas = []
    for box in horizontal_boxes:
        x, y, w, h = box
        spot_image = image[y:y+h, x:x+w]
        
        sum_value = np.sum(spot_image, axis=0)
        count_color_pixel = np.sum(np.where(spot_image > 0, 1, 0), axis=0)
        safe_count_color_pixel = np.where(count_color_pixel == 0, 1, count_color_pixel)
        average_value = (sum_value / safe_count_color_pixel).astype(int)
        average_value[count_color_pixel == 0] = 0
        
        peak_area = np.trapz(average_value)
        peak_areas.append(peak_area)
    return peak_areas

def _calculate_r_squared(peak_area: list, concentration_list: list, slope: float, intercept: float) -> float:
    y_hat = slope * np.array(concentration_list) + intercept
    y_bar = np.mean(peak_area)
    ss_tot = np.sum((peak_area - y_bar) ** 2)
    ss_res = np.sum((peak_area - y_hat) ** 2)
    return 1 - ss_res / ss_tot