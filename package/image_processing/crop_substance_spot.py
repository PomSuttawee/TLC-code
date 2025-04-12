import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MeanBackground
from scipy.signal import find_peaks


def crop_substance_spot_ingredient(image: np.ndarray) -> np.ndarray:
    assert image is not None, "Image is None"
    
    image_remove_background = remove_background(image)
    threshold_mask = create_threshold_mask(image_remove_background, mode='ingredient')
    assert np.sum(threshold_mask) > 0, "Threshold mask is empty"
    
    contours = detect_contours(threshold_mask)
    assert contours, "No contours found in the image"
    
    bounding_boxes = filter_contours(contours, image.shape, mode='ingredient')
    assert bounding_boxes, "No suitable contours found in the image"
    
    highest_concentration_bounding_boxes = get_highest_concentration_bounding_boxes(bounding_boxes)
    assert highest_concentration_bounding_boxes, "No highest concentration bounding boxes found"

    checked_overlapping_boxes = check_overlap(image, highest_concentration_bounding_boxes)
    assert checked_overlapping_boxes, "No checked overlapping boxes found"
    
    final_bounding_boxes = create_bounding_boxes_from_checked_boxes(bounding_boxes, checked_overlapping_boxes)
    assert final_bounding_boxes, "No final bounding boxes found"
    
    final_mask = create_mask_from_bounding_boxes(image.shape, final_bounding_boxes)
    assert np.sum(final_mask) > 0, "Final mask is empty"
    
    segmented_image = apply_mask(image, final_mask)
    assert segmented_image is not None, "Segmented image is None"

    
    # visualize_segmented_image(image, image_remove_background, threshold_mask, bounding_boxes, highest_concentration_bounding_boxes, checked_overlapping_boxes, final_bounding_boxes, final_mask, segmented_image)
    return segmented_image, final_bounding_boxes

def crop_substance_spot_mixture(image: np.ndarray) -> np.ndarray:
    assert image is not None, "Image is None"
    
    image_remove_background = remove_background(image)
    threshold_mask = create_threshold_mask(image_remove_background, 'mixture')
    assert np.sum(threshold_mask) > 0, "Threshold mask is empty"
    
    contours = detect_contours(threshold_mask)
    assert contours, "No contours found in the image"
    
    bounding_boxes = filter_contours(contours, image.shape)
    assert bounding_boxes, "No suitable contours found in the image"
    
    checked_overlapping_boxes = check_overlap(image, bounding_boxes)
    assert checked_overlapping_boxes, "No checked overlapping boxes found"
    
    sorted_boxes = sorted(checked_overlapping_boxes, key=lambda x: x[1], reverse=True)
    
    mask = create_mask_from_bounding_boxes(image.shape, sorted_boxes)
    assert np.sum(mask) > 0, "Mask is empty"
    
    segmented_image = apply_mask(image, mask)
    assert segmented_image is not None, "Segmented image is None"
    
    return segmented_image, sorted_boxes
    
def visualize_segmented_image(image: np.ndarray, image_remove_background: np.ndarray, threshold_mask: np.ndarray, all_boxes: list, highest_boxes: list, checked_boxes: list, final_boxes: list, final_mask: np.ndarray, segmented_image: np.ndarray) -> None:
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
    image_final_box = image.copy()
    for box in all_boxes:
        x, y, w, h = box
        cv2.rectangle(image_all_box, (x, y), (x+w, y+h), (0, 255, 0), 4)
    for box in highest_boxes:
        x, y, w, h = box
        cv2.rectangle(image_highest_concentration_box, (x, y), (x+w, y+h), (0, 255, 0), 4)
    for box in checked_boxes:
        x, y, w, h = box
        cv2.rectangle(image_checked_box, (x, y), (x+w, y+h), (0, 255, 0), 4)
    for horizontal_box in final_boxes:
        for box in horizontal_box:
            x, y, w, h = box
            cv2.rectangle(image_final_box, (x, y), (x+w, y+h), (0, 255, 0), 4)
    
    fig, ax = plt.subplots(3, 3, figsize=(18, 18))
    ax = ax.ravel()
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),                           ax[0].set_title("Original Image")
    ax[1].imshow(cv2.cvtColor(image_remove_background, cv2.COLOR_BGR2RGB), cmap='gray'),         ax[1].set_title("Background Removed")
    ax[2].imshow(threshold_mask, cmap='gray'),     ax[2].set_title("Threshold Mask")
    ax[3].imshow(cv2.cvtColor(image_all_box, cv2.COLOR_BGR2RGB)),                   ax[3].set_title("All Bounding Boxes")
    ax[4].imshow(cv2.cvtColor(image_highest_concentration_box, cv2.COLOR_BGR2RGB)), ax[4].set_title("Highest Concentration Bounding Boxes")
    ax[5].imshow(cv2.cvtColor(image_checked_box, cv2.COLOR_BGR2RGB)),               ax[5].set_title("Checked Overlapping Boxes")
    ax[6].imshow(cv2.cvtColor(image_final_box, cv2.COLOR_BGR2RGB)),                 ax[6].set_title("Final Bounding Boxes")
    ax[7].imshow(final_mask, cmap='gray'),         ax[7].set_title("Final Mask")
    ax[8].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)),                 ax[8].set_title("Segmented Image")
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

def create_threshold_mask(image: np.ndarray, mode: str = 'default') -> np.ndarray:
    """
    Create a binary threshold mask from the input image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Binary threshold mask.
    """
    iterations = 5 if mode == "ingredient" else 3
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel_size = max(5, min(image.shape[0] // 100, image.shape[1] // 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if mode == "ingredient":
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return threshold




############################################################################################################
# Contours Detection
############################################################################################################

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

def filter_contours(contours: list, image_shape: tuple, mode: str = 'default') -> list:
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
    if mode == 'ingredient':
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
    y_border_threshold = image_shape[0] * 0.025
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





############################################################################################################
# Bounding Boxes Processing
############################################################################################################

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
    
def create_bounding_boxes_from_checked_boxes(bounding_boxes: list, checked_boxes: list) -> list:
    """
    Create bounding boxes from the checked boxes.

    Args:
        bounding_boxes (list): List of bounding boxes.
        checked_boxes (list): List of checked boxes.

    Returns:
        list: List of final bounding boxes.
    """
    final_boxes = []
    y_coordinates = _get_y_coordinates(checked_boxes)
    x_coordinates = _get_x_coordinates(bounding_boxes)
    
    for y_start, y_end in y_coordinates:
        horizontal_boxes = []
        for x_start, x_end in x_coordinates:
            horizontal_boxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
        final_boxes.append(horizontal_boxes)
    return final_boxes

def _get_y_coordinates(boxes: list) -> list:
    """
    Get the y-coordinates of the input boxes.

    Args:
        boxes (list): List of bounding boxes.

    Returns:
        list: List of y-coordinates.
    """
    y_coordinates = [(box[1], box[1] + box[3]) for box in boxes]
    return sorted(y_coordinates, key=lambda x: x[0], reverse=True)

def _get_x_coordinates(boxes: list) -> list:
    """
    Get the x-coordinates of the input boxes.

    Args:
        boxes (list): List of bounding boxes.

    Returns:
        list: List of x-coordinates.
    """
    x_coordinates = [(box[0], box[0] + box[2]) for box in boxes]
    x_coordinates = _merge_intersect_x_coordinates(x_coordinates)
    return sorted(x_coordinates, key=lambda x: x[0])

def _merge_intersect_x_coordinates(x_coordinates: list) -> list:
    """
    Merge intersecting x-coordinates.

    Args:
        x_coordinates (list): List of x-coordinates.

    Returns:
        list: List of merged x-coordinates.
    """
    merged_coordinates = []
    sorted_x_coordinates = sorted(x_coordinates, key=lambda x: x[0])
    for x_start, x_end in sorted_x_coordinates:
        if not merged_coordinates:
            merged_coordinates.append([x_start, x_end])
        else:
            last_start, last_end = merged_coordinates[-1]
            if x_start > last_end or x_end < last_start:
                merged_coordinates.append([x_start, x_end])
            else:
                merged_coordinates[-1] = [min(x_start, last_start), max(x_end, last_end)]
    return merged_coordinates



def create_mask_from_bounding_boxes(image_shape: tuple, bounding_boxes: list) -> np.ndarray:
    """
    Create a binary mask from the input bounding boxes.

    Args:
        image_shape (tuple): Shape of the input image.
        bounding_boxes (list): List of bounding boxes.

    Returns:
        np.ndarray: Binary mask.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if isinstance(bounding_boxes[0], list):
        for boxes in bounding_boxes:
            for box in boxes:
                x, y, w, h = box
                mask = cv2.ellipse(img = mask,
                                    center = (x + w // 2, y + h // 2),
                                    axes = (w // 2, h // 2),
                                    angle = 0,
                                    startAngle = 0,
                                    endAngle = 360,
                                    color = 255,
                                    thickness = -1)
    else:
        for box in bounding_boxes:
            x, y, w, h = box
            mask = cv2.ellipse(img = mask,
                                center = (x + w // 2, y + h // 2),
                                axes = (w // 2, h // 2),
                                angle = 0,
                                startAngle = 0,
                                endAngle = 360,
                                color = 255,
                                thickness = -1)
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to the input image.

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Masked image.
    """
    assert image.shape[:2] == mask.shape, "Image and mask shape mismatch"
    return cv2.bitwise_and(image, image, mask=mask)