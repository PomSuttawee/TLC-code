import cv2
import numpy as np
from astropy.stats import SigmaClip
from photutils import Background2D, MeanBackground

def crop_using_something(image):
    """Process a single image through the entire segmentation pipeline."""
    l_channel = get_l_channel_image(image)
    remove_bg, background = remove_background(l_channel)

    threshold = threshold_image(remove_bg)
    contours = detect_contours(threshold)
    filtered_boxes = filter_contours(contours, image.shape)
    splitted_boxes = split_all_bounding_boxes(image, filtered_boxes)
    
    
    mask = create_mask_from_boxes(splitted_boxes, image.shape)
     
    return mask

def get_l_channel_image(image):
    """Convert image to LAB color space and extract L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]
    return l_channel

def remove_background(input_img):
    """Remove background using Background2D."""
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_mean = Background2D(input_img, (50, 50), filter_size=(3, 3), 
                            sigma_clip=sigma_clip, bkg_estimator=MeanBackground())
    return input_img - bkg_mean.background, bkg_mean.background

def threshold_image(image):
    """Apply thresholding and morphological operations."""
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    kernel_size = max(5, min(image.shape[0] // 100, image.shape[1] // 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=5)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return threshold

def detect_contours(threshold_img):
    """Detect contours in the thresholded image."""
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, image_shape):
    """Filter contours based on position and size."""
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    # Define borders
    x_border_threshold = image_shape[1] * 0.025
    y_border_threshold = image_shape[0] * 0.05
    x_border = (x_border_threshold, image_shape[1] - x_border_threshold)
    y_border = y_border_threshold

    # Filter by position and size
    min_area = 1500
    filtered_bounding_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        if x < x_border[0] or x + w > x_border[1] or y < y_border:
            continue
        if w * h < min_area:
            continue
        filtered_bounding_boxes.append(box)
        
    return filtered_bounding_boxes

def split_boxes_by_color(image, bounding_boxes):
    """Split bounding boxes based on color dissimilarity."""
    hsv_range = {
        "red": ([0, 50, 50], [30, 255, 255], [150, 50, 50], [180, 255, 255]),
        "green": ([31, 50, 50], [90, 255, 255]),
        "blue": ([91, 50, 50], [140, 255, 255]),
    }
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    
    boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        roi = h_channel[y:y+h, x:x+w]
        middle_x_roi = roi[:, w // 2]
    return boxes

def create_mask_from_boxes(image_shape, bounding_boxes):
    """Create a mask from the filtered bounding boxes."""
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
    for box in bounding_boxes:
        x, y, w, h = box
        mask = cv2.ellipse(mask, (x + w // 2, y + h // 2), 
                           (w // 2, h // 2), 0, 0, 360, 255, -1)
    
    return mask