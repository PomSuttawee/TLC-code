import numpy as np
import cv2
from scipy.signal import find_peaks

def crop_horizontal_lane_images(image: np.ndarray) -> list:
    contours = _get_contours(image)
    bounding_boxes = _get_bounding_boxes(contours)
    splitted_overlap_bounding_boxes = _split_overlap_bounding_boxes(image, bounding_boxes)
    grouped_bounding_boxes = _group_horizontal_bounding_boxes(image, splitted_overlap_bounding_boxes)
    
    convex_hulls = _create_convex_hulls(grouped_bounding_boxes)
    sorted_convex_hulls = sorted(convex_hulls, key=lambda x: cv2.boundingRect(x)[1])
    cropped_images = _crop_convex_hulls(image, sorted_convex_hulls)
    return cropped_images
    
def _get_contours(image: np.ndarray):
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _get_bounding_boxes(contours: list):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def _is_overlap_peak(image: np.ndarray):
    binarized_image = image.copy()
    binarized_image[binarized_image > 0] = 1
    y_sum_binarized = np.sum(binarized_image, axis=1)
    peak_index = find_peaks(y_sum_binarized)[0]
    return len(peak_index) > 1

def _split_bounding_box(image: np.ndarray, bounding_box: tuple):
    x, y, w, h = bounding_box
    cropped_image = image[y:y+h, x:x+w]
    
    binarized_image = cropped_image.copy()
    binarized_image[binarized_image > 0] = 1
    y_sum_binarized = np.sum(binarized_image, axis=1)
    inversed_y_sum_binarized = np.max(y_sum_binarized) - y_sum_binarized
    peak_index = int(find_peaks(inversed_y_sum_binarized)[0])
    
    return [(x, y, w, peak_index), (x, y+peak_index, w, h-peak_index)]

def _split_overlap_bounding_boxes(image: np.ndarray, bounding_boxes: list):
    non_overlap_bounding_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        cropped_image = image[y:y+h, x:x+w]
        
        if _is_overlap_peak(cropped_image):
            non_overlap_bounding_boxes += _split_bounding_box(image, box)
        else:
            non_overlap_bounding_boxes.append(box)

    return non_overlap_bounding_boxes

def _group_horizontal_bounding_boxes(image: np.ndarray, bounding_boxes: list):
    grouped_bounding_boxes = []
    for box in bounding_boxes:
        box_center_y = box[1] + (box[3] // 2)
        is_grouped = False
        
        if len(grouped_bounding_boxes) == 0:
            grouped_bounding_boxes.append([box])
            continue
        
        for grouped_box in grouped_bounding_boxes:
            grouped_box_center_y = grouped_box[0][1] + (grouped_box[0][3] // 2)
            threshold = max(box[3], grouped_box[0][3]) / 2
            
            if abs(box_center_y - grouped_box_center_y) < threshold:
                grouped_box.append(box)
                is_grouped = True
                break
            
        if not is_grouped:
            grouped_bounding_boxes.append([box])
            
    return grouped_bounding_boxes

def _create_convex_hulls(grouped_bounding_boxes: list):
    convex_hulls = []
    for group in grouped_bounding_boxes:
        pts = []
        for (x, y, w, h) in group:
            pts.append([x,   y])
            pts.append([x+w, y])
            pts.append([x+w, y+h])
            pts.append([x,   y+h])
        pts = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts)
        convex_hulls.append(hull)
    return convex_hulls

def _crop_convex_hulls(image: np.ndarray, convex_hulls: list):
    cropped_images = []
    for convex_hull in convex_hulls:
        mask = cv2.fillPoly(np.zeros_like(image), [convex_hull], (255, 255, 255))
        cropped_image = cv2.bitwise_and(image, mask)
        bounding_rect = cv2.boundingRect(cropped_image)
        cropped_images.append(cropped_image[bounding_rect[1]:bounding_rect[1]+bounding_rect[3], bounding_rect[0]:bounding_rect[0]+bounding_rect[2]])
    return cropped_images

def visualize_process(image: np.ndarray):
    contours = _get_contours(image)
    bounding_boxes = _get_bounding_boxes(contours)
    splitted_overlap_bounding_boxes = _split_overlap_bounding_boxes(image, bounding_boxes)
    grouped_bounding_boxes = _group_horizontal_bounding_boxes(image, splitted_overlap_bounding_boxes)
    convex_hulls = _create_convex_hulls(grouped_bounding_boxes)
    sorted_convex_hulls = sorted(convex_hulls, key=lambda x: cv2.boundingRect(x)[1])
    cropped_images = _crop_convex_hulls(image, sorted_convex_hulls)
    
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (255, 255, 255), 2)
    image_with_bounding_boxes = image.copy()
    image_with_splitted_overlap_bounding_boxes = image.copy()
    image_with_grouped_bounding_boxes = image.copy()
    image_With_convex_hulls = image.copy()
    
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image_with_bounding_boxes, (x, y), (x+w, y+h), (255, 255, 255), 2)
    for box in splitted_overlap_bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image_with_splitted_overlap_bounding_boxes, (x, y), (x+w, y+h), (255, 255, 255), 2)
    for i, group in enumerate(grouped_bounding_boxes):
        for box in group:
            x, y, w, h = box
            cv2.rectangle(image_with_grouped_bounding_boxes, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(image_with_grouped_bounding_boxes, str(i), (x+5, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    for hull in convex_hulls:
        cv2.drawContours(image_With_convex_hulls, [hull], -1, (255, 255, 255), 2)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs[0, 0].imshow(image, cmap='gray'), axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(image_with_contours, cmap='gray'), axs[0, 1].set_title("Contours")
    axs[0, 2].imshow(image_with_bounding_boxes, cmap='gray'), axs[0, 2].set_title("Bounding Boxes")
    axs[1, 0].imshow(image_with_splitted_overlap_bounding_boxes, cmap='gray'), axs[1, 0].set_title("Splitted Overlap Bounding Boxes")
    axs[1, 1].imshow(image_with_grouped_bounding_boxes, cmap='gray'), axs[1, 1].set_title("Grouped Bounding Boxes")
    axs[1, 2].imshow(image_With_convex_hulls, cmap='gray'), axs[1, 2].set_title("Convex Hulls")
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()