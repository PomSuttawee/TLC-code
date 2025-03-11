import cv2
import numpy as np

def _detect_contours(image: np.ndarray) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    return contours

def _remove_small_contours(contours: list, min_area: int) -> list:
    return [contour for contour in contours if cv2.contourArea(contour) > min_area]

def _union_contours(contours: list) -> tuple:
    x, y, w, h = cv2.boundingRect(contours[0])
    for contour in contours:
        x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(contour)
        x = min(x, x_temp)
        y = min(y, y_temp)
        w = max(w, w_temp)
        h = max(h, h_temp)
    return x, y, w, h

def _crop_by_contours(image: np.ndarray, contour: list) -> np.ndarray:
    x, y, w, h = contour
    return image[y:y+h, x:x+w]

def crop_to_paper(image: np.ndarray) -> np.ndarray:
    contours = _detect_contours(image)
    if not contours:
        raise ValueError("No contours found.")
    
    big_contours = _remove_small_contours(contours, 200)
    if not contours:
        raise ValueError(f"No contours with area > {200} found.")
    
    union_contours = _union_contours(big_contours)
    return _crop_by_contours(image, union_contours)

def visualize_process(image: np.ndarray):
    contours = _detect_contours(image)
    if not contours:
        raise ValueError("No contours found.")
    
    big_contours = _remove_small_contours(contours, 200)
    if not contours:
        raise ValueError(f"No contours with area > {200} found.")
    
    union_contours = _union_contours(big_contours)

    image_with_contours = image.copy()
    image_with_big_contours = image.copy()
    image_with_union_contours = image.copy()
    
    image_with_contours = cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)
    image_with_big_contours = cv2.drawContours(image_with_big_contours, big_contours, -1, (0, 255, 0), 3)
    image_with_union_contours = cv2.rectangle(image_with_union_contours, (union_contours[0], union_contours[1]), (union_contours[0] + union_contours[2], union_contours[1] + union_contours[3]), (0, 255, 0), 3)
    cropped_image = _crop_by_contours(image, union_contours)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(18, 18))
    axs[0, 0].imshow(image), axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(image_with_contours), axs[0, 1].set_title('Image with Contours')
    axs[0, 2].imshow(image_with_big_contours), axs[0, 2].set_title('Image with Big Contours')
    axs[1, 0].imshow(image_with_union_contours), axs[1, 0].set_title('Image with Union Contours')
    axs[1, 1].imshow(cropped_image), axs[1, 1].set_title('Cropped Image')
    
    for ax in axs.flat:
        ax.axis('off')
    plt.show()