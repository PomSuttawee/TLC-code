import cv2
import numpy as np
from package.image_processing.segmentation import util
from package.config import threshold_config

def segment_mixture(image: np.ndarray) -> np.ndarray:
    """
    Segment the mixture in the image using a predefined pipeline.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Segmented foreground image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy ndarray")
    pipeline = [
        _apply_gaussian_blur,
        _apply_clahe,
        _apply_adaptive_threshold_mixture,
        _apply_morph
    ]
    return _segment_image(image, pipeline)

def segment_ingredient(image: np.ndarray) -> np.ndarray:
    """
    Segment the ingredient in the image using a predefined pipeline.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Segmented foreground image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy ndarray")
    pipeline = [
        _apply_gaussian_blur,
        _apply_clahe,
        _apply_adaptive_threshold_calibration,
        _apply_morph
    ]
    return _segment_image(image, pipeline)

def crop_vertically(image: np.ndarray) -> list:
    """
    Crop the image vertically using the detected contours.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        list: List of cropped images.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy ndarray")
    list_contour = _get_contour(image, threshold_config.crop_vertically.min_area)
    list_box = _get_bounding_box_vertical(list_contour, image.shape[0])
    return _crop_by_bounding_box(image, list_box)

def crop_horizontally(image: np.ndarray) -> list:
    """
    Crop the image horizontally using the detected contours.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        list: List of cropped images.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy ndarray")
    list_contour = _get_contour(image, threshold_config.crop_horizontally.min_area)
    list_box = _get_bounding_box_horizontal(list_contour, image.shape[1])
    return _crop_by_bounding_box(image, list_box)

def _segment_image(image: np.ndarray, pipeline: list) -> np.ndarray:
    """
    Apply a series of processing functions to the image and return the segmented result.
    
    Args:
        image (np.ndarray): Input image.
        pipeline (list): List of processing functions.
    
    Returns:
        np.ndarray: Segmented image.
    """
    image_original = image.copy()
    foreground_mask = _apply_pipeline(image, pipeline)
    return _apply_mask(image_original, foreground_mask, 'and')

def _apply_pipeline(image: np.ndarray, pipeline: list) -> np.ndarray:
    """
    Apply a series of processing functions to the image.
    
    Args:
        image (np.ndarray): Input image.
        pipeline (list): List of processing functions.
    
    Returns:
        np.ndarray: Processed image.
    """
    for func in pipeline:
        image = func(image)
    return image
    
def _apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(
        src = image,
        ksize = threshold_config.gaussian_blur.kernel_size,
        sigmaX = threshold_config.gaussian_blur.sigma_x
    ) 
     
def _apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe_object = cv2.createCLAHE(
        clipLimit = threshold_config.clahe.clip_limit,
        tileGridSize = threshold_config.clahe.tile_grid_size
    )
    return clahe_object.apply(image)

def _apply_adaptive_threshold(image: np.ndarray, block_cnt: int) -> np.ndarray:
    if (image.shape[0] // block_cnt) % 2 == 1:
        block_size = image.shape[0] // block_cnt
    else:
        block_size = image.shape[0] // block_cnt + 1
    
    return cv2.adaptiveThreshold(
        src = image,
        maxValue = threshold_config.adaptive_threshold.max_value,
        adaptiveMethod = threshold_config.adaptive_threshold.adaptive_method,
        thresholdType = threshold_config.adaptive_threshold.threshold_type,
        blockSize = block_size,
        C = threshold_config.adaptive_threshold.constant
    )

def _apply_adaptive_threshold_calibration(image: np.ndarray) -> np.ndarray:
    return _apply_adaptive_threshold(image, threshold_config.adaptive_threshold.block_count_calibration)

def _apply_adaptive_threshold_mixture(image: np.ndarray) -> np.ndarray:
    return _apply_adaptive_threshold(image, threshold_config.adaptive_threshold.block_count_mixture)

def _apply_morph(image: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(
        src = image,
        op = cv2.MORPH_OPEN,
        kernel = cv2.getStructuringElement(
            shape = cv2.MORPH_ELLIPSE,
            ksize = threshold_config.morph.kernel_size
        )
    )
    
def _apply_mask(image: np.ndarray, mask: np.ndarray, operator: str) -> np.ndarray:
    if operator == 'and':
        return cv2.bitwise_and(image, mask)
    elif operator == 'or':
        return cv2.bitwise_or(image, mask)
    raise KeyError(f"Invalid operator: {operator}")

def _draw_contour(image: np.ndarray, list_contour: list) -> np.ndarray:
    new_image = image.copy()
    cv2.drawContours(
        image = new_image,
        contours = list_contour,
        contourIdx = threshold_config.draw_contours.contour_index,
        color = threshold_config.draw_contours.color,
        thickness = threshold_config.draw_contours.thickness,
        lineType = threshold_config.draw_contours.line_type
    )
    return new_image

def _draw_bounding_box(image: np.ndarray, list_box: list) -> np.ndarray:
    new_image = image.copy()
    for i, box in enumerate(list_box):
        x, y, w, h = box
        top_left_point = (x, y)
        bottom_right_point = (x + w, y + h)
        
        new_image = cv2.rectangle(
            img = new_image,
            pt1 = top_left_point,
            pt2 = bottom_right_point,
            color = threshold_config.draw_bounding_box.rectangle_color,
            thickness = threshold_config.draw_bounding_box.rectangle_thickness
        )
        
        new_image = cv2.putText(
            img = new_image,
            text = f"Peak {i + 1}",
            org = (10, y + h - 10),
            fontFace = threshold_config.draw_bounding_box.text_font,
            fontScale = threshold_config.draw_bounding_box.text_scale,
            color = threshold_config.draw_bounding_box.rectangle_color,
            thickness = threshold_config.draw_bounding_box.text_thickness
        )
    return new_image

def _get_contour(image: np.ndarray, min_area: int) -> list:
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_NONE
    list_contour, _ = cv2.findContours(image, mode, method)
    if min_area < 0:
        return list_contour
    return _remove_contour(list_contour, min_area)

def _remove_contour(list_contour: list, min_area: int) -> list:
    return [contour for contour in list_contour if cv2.contourArea(contour) > min_area]

def _get_bounding_box(list_contour: list, max_dim: int, vertical: bool) -> list:
    if vertical:
        list_box = [(x, 0, w, max_dim) for contour in list_contour for x, y, w, h in [cv2.boundingRect(contour)]]
        return sorted(util.groupBoundingBox(list_box), key=lambda x: x[0])
    else:
        list_box = [(0, y, max_dim, h) for contour in list_contour for x, y, w, h in [cv2.boundingRect(contour)]]
        return sorted(util.groupBoundingBox(list_box), key=lambda x: x[1])

def _get_bounding_box_vertical(list_contour: list, h_max: int) -> list:
    return _get_bounding_box(list_contour, h_max, vertical=True)

def _get_bounding_box_horizontal(list_contour: list, w_max: int) -> list:
    return _get_bounding_box(list_contour, w_max, vertical=False)

def _crop_by_bounding_box(image: np.ndarray, list_bounding_box: list) -> list:
    """
    Crop the image using the provided bounding boxes.
    
    Args:
        image (np.ndarray): Input image.
        list_bounding_box (list): List of bounding boxes.
    
    Returns:
        list: List of cropped images.
    """
    cropped_images = []
    for box in list_bounding_box:
        x, y, w, h = box
        cropped_images.append(image[y:y+h, x:x+w])
    return cropped_images