import numpy as np
import cv2

def crop_vertical_lane_images(image: np.ndarray):
    contours = _get_contours(image)
    bounding_boxes = _get_bounding_boxes_max_height(contours, image.shape[0])
    merged_bounding_boxes = _merge_overlap_bounding_boxes(bounding_boxes)
    sorted_bounding_boxes = sorted(merged_bounding_boxes, key=lambda x: x[0])
    return _crop_bounding_boxes(image, sorted_bounding_boxes)

def _get_contours(image: np.ndarray):
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _get_bounding_boxes_max_height(contours: list, max_height: int):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, 0, w, max_height))
    return bounding_boxes

def _crop_bounding_boxes(image: np.ndarray, bounding_boxes: list):
    cropped_images = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        cropped_images.append(image[y:y+h, x:x+w])
    return cropped_images

def _union(rect_1: tuple, rect_2: tuple) -> tuple:
    """
    Calculate the union of two rectangles.
    
    Args:
        rect_1 (tuple): The first rectangle (x, y, width, height).
        rect_2 (tuple): The second rectangle (x, y, width, height).
    
    Returns:
        tuple: The union of the two rectangles.
    """
    if not (isinstance(rect_1, tuple) and isinstance(rect_2, tuple)):
        raise ValueError("Input rectangles must be tuples")
    if len(rect_1) != 4 or len(rect_2) != 4:
        raise ValueError("Input rectangles must have four elements (x, y, width, height)")
    x = min(rect_1[0], rect_2[0])
    y = min(rect_1[1], rect_2[1])
    w = max(rect_1[0]+rect_1[2], rect_2[0]+rect_2[2]) - x
    h = max(rect_1[1]+rect_1[3], rect_2[1]+rect_2[3]) - y
    return (x, y, w, h)

def _intersection(rect_1: tuple, rect_2: tuple) -> tuple:
    """
    Calculate the intersection of two rectangles.
    
    Args:
        rect_1 (tuple): The first rectangle (x, y, width, height).
        rect_2 (tuple): The second rectangle (x, y, width, height).
    
    Returns:
        tuple: The intersection of the two rectangles, or None if they do not intersect.
    """
    if not (isinstance(rect_1, tuple) and isinstance(rect_2, tuple)):
        raise ValueError("Input rectangles must be tuples")
    if len(rect_1) != 4 or len(rect_2) != 4:
        raise ValueError("Input rectangles must have four elements (x, y, width, height)")
    x = max(rect_1[0], rect_2[0])
    y = max(rect_1[1], rect_2[1])
    w = min(rect_1[0]+rect_1[2], rect_2[0]+rect_2[2]) - x
    h = min(rect_1[1]+rect_1[3], rect_2[1]+rect_2[3]) - y
    if w < 0 or h < 0:
        return None
    return (x, y, w, h)

def _merge_overlap_bounding_boxes(rect_list: list) -> list:
    is_tested = [False for _ in range(len(rect_list))]
    final_rect = []
    i = 0

    while i < len(rect_list):
        if not is_tested[i]:
            j = i + 1
            while j < len(rect_list):
                if not is_tested[j]:
                    intersect = _intersection(rect_list[i], rect_list[j])
                    if intersect:
                        rect_list[i] = _union(rect_list[i], rect_list[j])
                        is_tested[j] = True
                        j = i
                j += 1
            final_rect.append(rect_list[i])
        i += 1
    return final_rect