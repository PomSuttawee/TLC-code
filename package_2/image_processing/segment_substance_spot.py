import cv2
import numpy as np
from typing import Tuple

# Constants
STD_DEVIATION_MULTIPLIER = 2.0
KERNEL_SIZE = (7, 7)
OPEN_ITERATIONS = 2
CLOSE_ITERATIONS = 1

def segment_substance_spot(image: np.ndarray) -> np.ndarray:
    """
    Segment substance spots in an image.
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple containing (mask, masked_image)
    """
    mask = create_foreground_mask(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def create_foreground_mask(image: np.ndarray, tiles: int = 1) -> np.ndarray:
    """
    Create a foreground mask by optionally splitting the image into an
    NxN grid of tiles and processing each tile separately.

    Args:
        image: Input BGR image
        tiles: 1 for a global mask, 2 for 2x2 tiling, etc.
        
    Returns:
        Foreground mask (uint8)
    """
    # If tiles == 1, process the entire image globally
    if tiles == 1:
        return _create_mask_for_patch(image)

    # Otherwise, split into an NxN grid and process each tile
    h, w = image.shape[:2]
    tile_h, tile_w = h // tiles, w // tiles
    
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for row in range(tiles):
        for col in range(tiles):
            y_start = row * tile_h
            y_end = (row + 1) * tile_h if row < tiles - 1 else h
            x_start = col * tile_w
            x_end = (col + 1) * tile_w if col < tiles - 1 else w
            
            tile = image[y_start:y_end, x_start:x_end]
            tile_mask = _create_mask_for_patch(tile)
            final_mask[y_start:y_end, x_start:x_end] = tile_mask
            
    return final_mask

def _create_mask_for_patch(patch: np.ndarray) -> np.ndarray:
    """
    Create a foreground mask for one patch of the image using Lab color space analysis.
    
    This function converts the image to Lab color space and identifies foreground pixels
    by finding colors that deviate significantly from the median values in a and b channels.
    
    Args:
        patch: Input BGR image patch
        
    Returns:
        Foreground mask (uint8)
    """
    # Convert to Lab color space
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
    
    # Calculate statistics for a and b channels
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    med_a, std_a = np.median(a_channel), np.std(a_channel)
    med_b, std_b = np.median(b_channel), np.std(b_channel)
    
    # Calculate thresholds
    a_thresholds = (
        med_a - STD_DEVIATION_MULTIPLIER * std_a,
        med_a + STD_DEVIATION_MULTIPLIER * std_a
    )
    b_thresholds = (
        med_b - STD_DEVIATION_MULTIPLIER * std_b,
        med_b + STD_DEVIATION_MULTIPLIER * std_b
    )
    
    # Create masks for each channel
    bg_mask_a = cv2.inRange(a_channel, *a_thresholds)
    bg_mask_b = cv2.inRange(b_channel, *b_thresholds)
    
    # Invert to get foreground masks
    fg_mask_a = cv2.bitwise_not(bg_mask_a)
    fg_mask_b = cv2.bitwise_not(bg_mask_b)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
    
    for mask in [fg_mask_a, fg_mask_b]:
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                        dst=mask, iterations=OPEN_ITERATIONS)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                        dst=mask, iterations=CLOSE_ITERATIONS)
    
    # Combine masks
    fg_mask_combined = cv2.bitwise_or(fg_mask_a, fg_mask_b)
    return fg_mask_combined