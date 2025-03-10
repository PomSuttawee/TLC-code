import cv2
import numpy as np

def _is_valid_length(line):
    x1, y1, x2, y2 = line
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return line_length > 50

def _is_valid_angle(line):
    x1, y1, x2, y2 = line
    line_angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
    return line_angle <= 5 or line_angle >= (180 - 5)

def _is_valid_y_coordinate(image, line):
    x1, y1, x2, y2 = line
    valid_range = [0.05 * image.shape[0], 0.95 * image.shape[0]]
    return y1 >= valid_range[0] and y2 >= valid_range[0] and y1 <= valid_range[1] and y2 <= valid_range[1]

def _is_valid_line(image, line):
    return _is_valid_length(line) and _is_valid_angle(line) and _is_valid_y_coordinate(image, line)

def filter_valid_lines(image: np.ndarray, lines: list) -> list:
    return [line for line in lines if _is_valid_line(image, line)]

def _detect_line_lsd(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines, _, _, _ = lsd.detect(gray)
    return [line[0] for line in lines]

def _filter_top_and_bottom_lines(image: np.ndarray, lines: list) -> list:
    top_lines = []
    bottom_lines = []
    
    mid_point = image.shape[0] // 2
    for line in lines:
        x1, y1, x2, y2 = line
        if y1 < mid_point and y2 < mid_point:
            top_lines.append(line)
        elif y1 > mid_point and y2 > mid_point:
            bottom_lines.append(line)
    return top_lines, bottom_lines

def _calculate_average_y(lines: list) -> tuple:
    total_y = 0
    for line in lines:
        x1, y1, x2, y2 = line
        total_y += y1 + y2
    average_y = total_y / (len(lines) * 2)
    return average_y

def _crop_by_y_coordinate(image: np.ndarray, y_coordinate_range: tuple) -> np.ndarray:
    return image[int(y_coordinate_range[0]):int(y_coordinate_range[1]), :]

def crop_solvent_front_and_origin(image: np.ndarray) -> np.ndarray:
    lines = _detect_line_lsd(image)
    if lines is None:
        raise ValueError("No lines found in the image.")
    
    valid_lines = filter_valid_lines(image, lines)
    if not valid_lines:
        raise ValueError("No valid lines found in the image.")
    
    top_lines, bottom_lines = _filter_top_and_bottom_lines(image, valid_lines)
    if not top_lines:
        raise ValueError("No top lines found in the image.")
    elif not bottom_lines:
        raise ValueError("No bottom lines found in the image.")
    
    average_top_line = _calculate_average_y(top_lines)
    average_bottom_line = _calculate_average_y(bottom_lines)
    cropped_image = _crop_by_y_coordinate(image, (average_top_line, average_bottom_line))
     
    return cropped_image