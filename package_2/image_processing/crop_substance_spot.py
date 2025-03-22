import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MeanBackground
from scipy.signal import find_peaks


def crop_substance_spot(image):
    assert image is not None, "Image is None"
    
    image_remove_background = remove_background(image)
    threshold_mask = create_threshold_mask(image_remove_background)
    assert np.sum(threshold_mask) > 0, "Threshold mask is empty"
    
    contours = detect_contours(threshold_mask)
    assert contours, "No contours found in the image"
    
    bounding_boxes = filter_contours(contours, image.shape)
    assert bounding_boxes, "No suitable contours found in the image"
    
    highest_concentration_bounding_boxes = get_highest_concentration_bounding_boxes(bounding_boxes)
    assert highest_concentration_bounding_boxes, "No highest concentration bounding boxes found"

    checked_overlapping_boxes = check_overlap(image, highest_concentration_bounding_boxes)
    assert checked_overlapping_boxes, "No checked overlapping boxes found"
    
    
    
    visualize_segmented_image(image, image_remove_background, threshold_mask, bounding_boxes, highest_concentration_bounding_boxes, checked_overlapping_boxes)
    return None
    
    return segmented_image, bounding_boxes

def visualize_segmented_image(image: np.ndarray, image_remove_background: np.ndarray, threshold_mask: np.ndarray, all_boxes: list, highest_boxes: list, checked_boxes: list) -> None:
    """
    Visualize the segmented image with detected bounding boxes.

    Args:
        image (np.ndarray): Original input image.
        image_remove_background (np.ndarray): Image with the background removed.
        threshold_mask (np.ndarray): Binary threshold mask.
        bounding_boxes (list): List of detected bounding boxes.
    """
    image_all_box = image.copy()
    image_highest_concentration_box = image.copy()
    image_checked_box = image.copy()
    for box in all_boxes:
        x, y, w, h = box
        cv2.rectangle(image_all_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for box in highest_boxes:
        x, y, w, h = box
        cv2.rectangle(image_highest_concentration_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for box in checked_boxes:
        x, y, w, h = box
        cv2.rectangle(image_checked_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    fig, ax = plt.subplots(2, 3, figsize=(18, 18))
    ax = ax.ravel()
    ax[0].imshow(image),                           ax[0].set_title("Original Image")
    ax[1].imshow(image_remove_background),         ax[1].set_title("Background Removed")
    ax[2].imshow(threshold_mask, cmap='gray'),     ax[2].set_title("Threshold Mask")
    ax[3].imshow(image_all_box),                   ax[3].set_title("All Bounding Boxes")
    ax[4].imshow(image_highest_concentration_box), ax[4].set_title("Highest Concentration Bounding Boxes")
    ax[5].imshow(image_checked_box),               ax[5].set_title("Checked Overlapping Boxes")
    for a in ax:
        a.axis("off")
    plt.show()

def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Remove the background from the input image using a sigma-clipped mean background estimation.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image with the background removed.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MeanBackground(sigma_clip=sigma_clip)
    bkg = Background2D(
        data = l,
        box_size = (50, 50),
        filter_size = (3, 3),
        bkg_estimator = bkg_estimator
        )
    return l - bkg.background

def create_threshold_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary threshold mask from the input image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Binary threshold mask.
    """
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel_size = max(5, min(image.shape[0] // 100, image.shape[1] // 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=5)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=3)
    return threshold

def detect_contours(image: np.ndarray):
    """
    Detect contours in the input image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        list: List of detected contours.
    """
    contours, _ = cv2.findContours(
        image = image,
        mode = cv2.RETR_EXTERNAL,
        method = cv2.CHAIN_APPROX_SIMPLE
        )
    return contours

def filter_contours(contours: list, image_shape: tuple) -> list:
    """
    Filter detected contours based on size and shape.

    Args:
        contours (list): List of detected contours.
        image_shape (tuple): Shape of the input image.

    Returns:
        list: Filtered list of contours.
    """
    filtered_contours = _remove_small_contours(contours, image_shape)
    filtered_contours = _filter_non_border_contours(filtered_contours, image_shape)
    filtered_contours = _remove_right_most_vertical_lane(filtered_contours)
    return filtered_contours

def _remove_small_contours(contours: list, image_shape: tuple) -> list:
    """
    Filter out contours that are too big.

    Args:
        contours (list): List of detected contours.
        image_shape (tuple): Shape of the input image.

    Returns:
        list: Filtered list of contours.
    """
    min_area = image_shape[0] * image_shape[1] * 0.001
    return [contour for contour in contours if cv2.contourArea(contour) > min_area]

def _filter_non_border_contours(contours: list, image_shape: tuple) -> list:
    """
    Filter out contours that are too close to the image borders.

    Args:
        contours (list): List of detected contours.
        image_shape (tuple): Shape of the input image.

    Returns:
        list: Filtered list of contours.
    """
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    x_border_threshold = image_shape[1] * 0.025
    y_border_threshold = image_shape[0] * 0.05
    x_border = (x_border_threshold, image_shape[1] - x_border_threshold)
    y_border = y_border_threshold

    filtered_bounding_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        if x < x_border[0] or x + w > x_border[1] or y < y_border:
            continue
        filtered_bounding_boxes.append(box)
    return filtered_bounding_boxes

def _remove_right_most_vertical_lane(boxes: list) -> list:
    """
    Remove the right-most vertical lane from the list of detected contours.

    Args:
        contours (list): List of detected contours.

    Returns:
        list: Filtered list of contours.
    """
    sorted_boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    remove_range = (sorted_boxes[0][0], sorted_boxes[0][0] + sorted_boxes[0][2])
    for i, box in enumerate(sorted_boxes[1:]):
        if _intersect(box, remove_range):
            continue
        else:
            return sorted_boxes[i+1:]

def _intersect(box: tuple, remove_range: tuple) -> bool:
    """
    Check if the input box intersects with the remove range.

    Args:
        box (tuple): Bounding box coordinates (x, y, w, h).
        remove_range (tuple): Range of the right-most vertical lane.

    Returns:
        bool: True if the box intersects with the remove range, False otherwise.
    """
    x, y, w, h = box
    return x < remove_range[1] and x + w > remove_range[0]

def get_highest_concentration_bounding_boxes(bounding_boxes: list) -> list:
    """
    Get the bounding boxes with the highest concentration of substance spots.

    Args:
        bounding_boxes (list): List of bounding boxes.

    Returns:
        list: List of bounding boxes with the highest concentration.
    """
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x[0], reverse=True)
    center_seed = sorted_boxes[0][0] + sorted_boxes[0][2] // 2
    for i, box in enumerate(sorted_boxes[1:]):
        if box[0] < center_seed < box[0] + box[2]:
            continue
        else:
            return sorted_boxes[:i+1]

def check_overlap(image: np.ndarray, bounding_boxes: list) -> list:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    l_channel = cv2.GaussianBlur(l_channel, (5, 5), 0)
    
    splitted_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        cropped_l_channel = l_channel[y:y+h, x:x+w]
        if _is_overlap(cropped_l_channel):
            splitted_boxes.extend(_split_box(cropped_l_channel, box))
        else:
            splitted_boxes.append(box)
    return splitted_boxes

def _is_overlap(image: np.ndarray) -> bool:
    """
    Check if the input image contains overlapping objects.

    Args:
        image (np.ndarray): Input image.

    Returns:
        bool: True if the image contains overlapping objects, False otherwise.
    """
    image = image / 255
    image = 1 - image
    y_sum = np.sum(image, axis=1)
    peak_index = find_peaks(y_sum, prominence=1)[0]
    return len(peak_index) > 1

def _split_box(image: np.ndarray, box: tuple) -> list:
    """
    Split the input bounding box into two separate boxes.

    Args:
        image (np.ndarray): Input image.
        box (tuple): Bounding box coordinates (x, y, w, h).

    Returns:
        list: List of two bounding boxes.
    """
    image = image / 255
    y_sum = np.sum(image, axis=1)
    minima_index = find_peaks(y_sum, prominence=1)[0]
    
    x, y, w, h = box
    if len(minima_index) == 1:
        return [(x, y, w, minima_index[0]), (x, y + minima_index[0], w, h - minima_index[0])]
    elif len(minima_index) == 2:
        return [(x, y, w, minima_index[0]),
                (x, y + minima_index[0], w, minima_index[1] - minima_index[0]),
                (x, y + minima_index[1], w, h - minima_index[1])]
    else:
        return [box]
    
    
    