import logging
from typing import List, Tuple
import numpy as np
import cv2
from package.config import crop_config, line_detection_config

def crop_to_largest_contour(image: np.ndarray) -> np.ndarray:
    """
    Crops an image to the largest detected quadrilateral contour.

    This function processes the input image to detect contours, identifies 
    the largest quadrilateral contour, and crops the image to fit the detected
    contour. If no contours are found or no suitable quadrilateral contour 
    exists, the function returns `None`.

    Args:
        image (numpy.ndarray): Input image in RGB format.

    Returns:
        numpy.ndarray: Cropped image containing the largest quadrilateral contour, 
                       or `None` if no suitable contour is found.
    """
    log = logging.getLogger('Crop')
    try:
        log.debug("Applying Gaussian blur to the image.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        blur = cv2.GaussianBlur(
            src=gray,
            ksize=crop_config.gaussian_blur.kernel_size,
            sigmaX=crop_config.gaussian_blur.sigma_x
        )

        log.debug("Applying threshold to the blurred image.")
        _, threshold = cv2.threshold(
            src=blur,
            thresh=crop_config.thresholding.threshold_value,
            maxval=crop_config.thresholding.max_value,
            type=crop_config.thresholding.type
        )

        log.debug("Finding contours in the thresholded image.")
        contours, _ = cv2.findContours(
            image=threshold, 
            mode=crop_config.contours.contours_retrieval_mode, 
            method=crop_config.contours.contours_approximation_method
        )

        log.info(f"Number of contours found: {len(contours)}")
        if not contours:
            log.warning("No contours found in the image.")
            return None

        # Find the largest rectangular contour
        largest_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if area > max_area and len(approx) == 4:
                largest_contour = approx
                max_area = area
                log.debug(f"New largest quadrilateral contour found with area: {area}")
                # Explanation: Update largest_contour if a larger quadrilateral is found

        # If a contour is found, crop the image
        if largest_contour is None:
            log.warning("No suitable quadrilateral contour found in the image.")
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        log.debug(f"Cropping the image at x={x}, y={y}, w={w}, h={h}.")
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

    except cv2.error as e:
        log.error(f"OpenCV error during cropping: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error during cropping: {e}")
        return None

def _detect_edge(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in an image using the Canny edge detector.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image with detected edges.
    """
    if line_detection_config.canny.auto_threshold_ratio:
        median = np.median(image)
        lower = max(0, int(median * line_detection_config.canny.auto_threshold_ratio[0]))
        upper = min(255, int(median * line_detection_config.canny.auto_threshold_ratio[1]))
    else:
        lower = line_detection_config.canny.lower_threshold
        upper = line_detection_config.canny.upper_threshold
    
    return cv2.Canny(
        image = image,
        threshold1 = lower, 
        threshold2 = upper,
        L2gradient = line_detection_config.canny.use_l2_gradient
        )

def _calculate_average_lines(lines: list) -> Tuple[int, int]:
    """
    Calculate the average lines from the detected lines.

    Args:
        lines (list): Detected lines.

    Returns:
        list: Average lines.
    """
    if not lines:
        return (None, None)
    sum_rho = 0
    for rho, theta in lines:
        sum_rho += rho
    average_rho = sum_rho / len(lines)
    
    sum_lower_rho = 0
    sum_upper_rho = 0
    count_lower_rho = 0
    count_upper_rho = 0
    for rho, theta in lines:
        if rho > average_rho:
            sum_upper_rho += rho
            count_upper_rho += 1
        else:
            sum_lower_rho += rho
            count_lower_rho += 1  
    average_upper_rho = int(sum_upper_rho / count_upper_rho) if count_upper_rho else None
    average_lower_rho = int(sum_lower_rho / count_lower_rho) if count_lower_rho else None
    return (average_upper_rho, average_lower_rho)

def _filter_houghlines(lines: np.ndarray, image_height: int) -> List[List[float]]:
    """
    Filter lines based on rho distance from image borders.

    Args:
        lines (np.ndarray): Detected lines.
        image_height (int): Height of the image.

    Returns:
        List[List[float]]: Filtered lines.
    """
    min_rho = line_detection_config.filter.min_border_distance
    max_rho = image_height - min_rho
    return [[rho, theta] for rho, theta in lines[:, 0] if min_rho < rho < max_rho]

def _crop_image_using_lines(image: np.ndarray, lines: Tuple[int, int]) -> np.ndarray:
    """
    Crop the image using the detected lines.

    Args:
        image (np.ndarray): Input image.
        lines: Detected lines.

    Returns:
        np.ndarray: Cropped image.
    """
    if not lines:
        return image

    top_y, bottom_y = _calculate_average_lines(lines)
    cropped_image = image[bottom_y:top_y, :]
    return cropped_image

def detect_and_crop_houghlines(image: np.ndarray) -> np.ndarray:
    """
    Detect lines in an image using the Hough Line Transform.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image with detected lines drawn.
    """
    log = logging.getLogger('Line Detection')
    log.info(f'Detecting and cropping lines in the image')
    
    log.debug(f'Detecting edges')
    edges = _detect_edge(image)
    
    log.debug(f'Finding lines')
    lines = cv2.HoughLines(
        image = edges,
        rho = line_detection_config.hough_lines.rho,
        theta = line_detection_config.hough_lines.theta,
        threshold = int(image.shape[1] * line_detection_config.hough_lines.adaptive_threshold),
        min_theta = line_detection_config.hough_lines.min_theta,
        max_theta = line_detection_config.hough_lines.max_theta
        )
    
    if lines is None:
        log.info(f'No lines found')
        return image.copy()
    
    log.debug(f'Filtering lines')
    filtered_lines = _filter_houghlines(lines = lines, image_height = image.shape[0])
    # images_with_lines = _draw_houghlines(image, filtered_lines)
    
    log.debug(f'Cropping image')
    cropped_image = _crop_image_using_lines(image, filtered_lines)
    return cropped_image