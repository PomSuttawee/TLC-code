import cv2
import numpy as np

def segment_substance_spot(image: np.ndarray) -> np.ndarray:
    mask = create_foreground_mask(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def _create_mask_for_patch(patch: np.ndarray) -> np.ndarray:
    """
    Create a foreground mask for one patch of the image (same logic
    as your original _create_foreground_mask).
    """
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
    
    med_a = np.median(lab[:, :, 1])
    med_b = np.median(lab[:, :, 2])
    std_a = np.std(lab[:, :, 1])
    std_b = np.std(lab[:, :, 2])
    
    a_lower = med_a - 2 * std_a
    a_upper = med_a + 2 * std_a
    b_lower = med_b - 2 * std_b
    b_upper = med_b + 2 * std_b
    
    # Note: inRange expects numeric bounds. If your image or patch is uint8,
    # be mindful that floats will get truncated. Usually it's fine,
    # but you may want to cast to int if necessary.
    bg_mask_a = cv2.inRange(lab[:, :, 1], a_lower, a_upper)
    bg_mask_b = cv2.inRange(lab[:, :, 2], b_lower, b_upper)
    
    fg_mask_a = cv2.bitwise_not(bg_mask_a)
    fg_mask_b = cv2.bitwise_not(bg_mask_b)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask_a = cv2.morphologyEx(fg_mask_a, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask_a = cv2.morphologyEx(fg_mask_a, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask_b = cv2.morphologyEx(fg_mask_b, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask_b = cv2.morphologyEx(fg_mask_b, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    fg_mask_ab = cv2.bitwise_or(fg_mask_a, fg_mask_b)
    return fg_mask_ab

def create_foreground_mask(image: np.ndarray, tiles: int = 1) -> np.ndarray:
    """
    Create a foreground mask by optionally splitting the image into an
    NxN grid of tiles (where N=tiles) and applying the foreground mask
    creation logic to each tile separately.

    :param image: Input BGR image.
    :param tiles: 1 for a global mask, 2 for 2x2 tiling, etc.
    :return: Foreground mask (uint8).
    """
    # If tiles == 1, simply do the entire image globally
    if tiles == 1:
        return _create_mask_for_patch(image)

    # Otherwise, split into an NxN grid
    h, w = image.shape[:2]
    tile_h = h // tiles
    tile_w = w // tiles
    
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Iterate over each tile
    for row in range(tiles):
        for col in range(tiles):
            # Compute the boundaries of the current tile
            y_start = row * tile_h
            y_end = (row + 1) * tile_h if row < tiles - 1 else h
            x_start = col * tile_w
            x_end = (col + 1) * tile_w if col < tiles - 1 else w
            
            tile = image[y_start:y_end, x_start:x_end]
            
            # Apply the mask creation logic to this tile
            tile_mask = _create_mask_for_patch(tile)
            
            # Place the tile's mask into the final mask
            final_mask[y_start:y_end, x_start:x_end] = tile_mask
    
    return final_mask
