import numpy as np
import cv2
from scipy.signal import find_peaks

def extract_data(image: np.ndarray):
    contours = _get_contours(image)
    bounding_boxes = _get_bounding_boxes(contours)
    non_overlap_bounding_boxes = _split_overlap_bounding_boxes(image, bounding_boxes)
    sorted_bounding_boxes = _sort_bounding_boxes(non_overlap_bounding_boxes, sort_by = 'y')
    data = _calculate_rf_and_peak_area(image, sorted_bounding_boxes)
    return data
    
def _get_contours(image: np.ndarray):
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _get_bounding_boxes(contours: list):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def _is_overlap_peak(image: np.ndarray):
    binarized_image = image.copy()
    binarized_image[binarized_image > 0] = 1
    y_sum_binarized = np.sum(binarized_image, axis=1)
    peak_index = find_peaks(y_sum_binarized)[0]
    return len(peak_index) > 1

def _split_bounding_box(image: np.ndarray, bounding_box: tuple):
    x, y, w, h = bounding_box
    cropped_image = image[y:y+h, x:x+w]
    
    binarized_image = cropped_image.copy()
    binarized_image[binarized_image > 0] = 1
    y_sum_binarized = np.sum(binarized_image, axis=1)
    inversed_y_sum_binarized = np.max(y_sum_binarized) - y_sum_binarized
    peak_index = int(find_peaks(inversed_y_sum_binarized)[0])
    
    return [(x, y, w, peak_index), (x, y+peak_index, w, h-peak_index)]

def _split_overlap_bounding_boxes(image: np.ndarray, bounding_boxes: list):
    non_overlap_bounding_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        cropped_image = image[y:y+h, x:x+w]
        
        if _is_overlap_peak(cropped_image):
            non_overlap_bounding_boxes += _split_bounding_box(image, box)
        else:
            non_overlap_bounding_boxes.append(box)

    return non_overlap_bounding_boxes

def _sort_bounding_boxes(bounding_boxes: list, sort_by: str):
    if sort_by == 'y':
        return sorted(bounding_boxes, key=lambda x: x[1])
    elif sort_by == 'x':
        return sorted(bounding_boxes, key=lambda x: x[0])
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}. Must be 'x' or 'y'.")

def _crop_by_bounding_box(image: np.ndarray, bounding_box: tuple):
    x, y, w, h = bounding_box
    return image[y:y+h, x:x+w]

def _calculate_rf(y_coordinate: int, image_height: int) -> float:
    return (image_height - y_coordinate) / image_height

def _calculate_peak_area(spot_image: np.ndarray) -> float:
    average_value = np.sum(spot_image, axis=0)
    count_color_pixel = np.sum(np.where(spot_image > 0, 1, 0), axis=0)
    safe_count_color_pixel = np.where(count_color_pixel == 0, 1, count_color_pixel)
    average_value = (255 - (average_value / safe_count_color_pixel)).astype(int)
    average_value[count_color_pixel == 0] = 0
    peak_area = np.trapz(average_value)
    return peak_area

def _calculate_rf_and_peak_area(image: np.ndarray, bounding_boxes: list):
    data_dict = {}
    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box
        spot_center_on_y = y + h // 2
        
        spot_image = _crop_by_bounding_box(image, box)
        rf = _calculate_rf(spot_center_on_y, image.shape[0])
        peak_area = _calculate_peak_area(spot_image)

        data_dict[i] = {
            'spot_image': spot_image,
            'rf': rf,
            'peak_area': peak_area
        }
    return data_dict