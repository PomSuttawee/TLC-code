import cv2
import numpy as np

def _convert_RGB_to_HSV(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def segment_hsv_threshold_range(image: np.ndarray) -> np.ndarray:
    hsv = _convert_RGB_to_HSV(image)
    lower_threshold = np.array([0, 40, 50])
    upper_threshold = np.array([360, 255, 255])
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
    result = cv2.bitwise_and(image, image, mask=morph)
    return result

def visualize_process(image: np.ndarray):
    hsv = _convert_RGB_to_HSV(image)
    lower_threshold = np.array([0, 40, 50])
    upper_threshold = np.array([360, 255, 255])
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
    result = cv2.bitwise_and(image, image, mask=morph)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    axs[0, 0].imshow(image[:, :, 0], cmap='gray'), axs[0, 0].set_title("Red Channel")
    axs[0, 1].imshow(image[:, :, 1], cmap='gray'), axs[0, 1].set_title("Green Channel")
    axs[0, 2].imshow(image[:, :, 2], cmap='gray'), axs[0, 2].set_title("Blue Channel")
    axs[1, 0].imshow(hsv[:, :, 0], cmap='gray'), axs[1, 0].set_title("Hue Channel")
    axs[1, 1].imshow(hsv[:, :, 1], cmap='gray'), axs[1, 1].set_title("Saturation Channel")
    axs[1, 2].imshow(hsv[:, :, 2], cmap='gray'), axs[1, 2].set_title("Value Channel")
    axs[2, 0].imshow(mask, cmap='gray'), axs[2, 0].set_title("Mask")
    axs[2, 1].imshow(morph, cmap='gray'), axs[2, 1].set_title("Morphological Transformation")
    axs[2, 2].imshow(result), axs[2, 2].set_title("Segmented Image")
    for ax in axs.flat:
        ax.axis('off')
    plt.show()
    