import numpy as np

def extract_data(image: np.ndarray, bounding_boxes: list) -> dict:
    assert isinstance(image, np.ndarray), "Image must be a numpy array."
    assert isinstance(bounding_boxes, list), "Bounding boxes must be a list."
    
    rf_data = calculate_rf(image, bounding_boxes)
    peak_area_data = calculate_peak_area(image, bounding_boxes)
    data = {}
    for i in range(len(bounding_boxes)):
        data[f"H{i+1}"] = {"rf": rf_data[i], "peak_area": peak_area_data[i]}
    return data

def calculate_rf(image: np.ndarray, bounding_boxes: list) -> list:
    total_distance = image.shape[0]
    rf_data = []
    for boxes in bounding_boxes:
        center_y = boxes[1] + boxes[3] // 2
        rf = (total_distance - center_y) / total_distance
        rf_data.append(rf)
    return rf_data

def calculate_peak_area(image: np.ndarray, bounding_boxes: list) -> list:
    peak_areas = []
    for box in bounding_boxes:
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