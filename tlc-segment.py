from package.image_processing.general import io, crop

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # image = cv2.resize(image, (512, 512))  # Optional: Resize for consistency
    
    # Denoise using Non-Local Means
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Convert to LAB color space for better color separation
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    return lab

def kmeans_segmentation(lab_image, num_clusters=3):
    # Reshape to 2D array of pixels
    pixels = lab_image.reshape(-1, 3)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(lab_image.shape[:2])
    
    # Find the cluster with the highest intensity (assuming spots are bright)
    cluster_intensity = [np.mean(lab_image[labels == i]) for i in range(num_clusters)]
    spot_cluster = np.argmax(cluster_intensity)
    
    # Create a binary mask
    mask = np.uint8(labels == spot_cluster) * 255
    return mask

def refine_mask(mask):
    # Morphological closing to fill small holes
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise
    cleaned = cv2.connectedComponentsWithStats(closed, 4, cv2.CV_32S)
    num_labels, label_map = cleaned[0], cleaned[1]
    max_label = 1 + np.argmax(cleaned[2][1:, cv2.CC_STAT_AREA])
    refined_mask = np.uint8(label_map == max_label) * 255
    return refined_mask

def superpixel_refinement(image, mask):
    # Convert image to float
    image_float = img_as_float(image)
    
    # Compute SLIC superpixels
    segments = slic(image_float, n_segments=100, compactness=10)
    
    # Mask superpixels overlapping with the initial mask
    superpixel_mask = np.zeros_like(mask)
    for sp in np.unique(segments):
        if np.mean(mask[segments == sp]) > 0.5:  # Threshold overlap
            superpixel_mask[segments == sp] = 255
    return superpixel_mask

def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def segment_tlc_spot(image_path):
    lab_image = preprocess_image(image_path)
    mask = kmeans_segmentation(lab_image)
    refined_mask = refine_mask(mask)
    # Optional: refined_mask = superpixel_refinement(lab_image, refined_mask)
    contours = extract_contours(refined_mask)
    
    # Visualize
    original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(original), plt.title('Original')
    plt.subplot(122), plt.imshow(refined_mask, cmap='gray'), plt.title('Segmented Mask')
    plt.show()
    
    return refined_mask, contours

def normalize_image(image: np.ndarray):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def convert_and_show_image_in_various_color_space(image):
    cropped_image = crop.detect_and_crop_houghlines(crop.crop_to_largest_contour(image))
    normalized_image = cv2.cvtColor(normalize_image(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB)
    lab_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2LAB)
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    ycbcr_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2YCrCb)
    
    plt.figure(figsize=(15, 8))
    plt.subplot(5, 3, 1), plt.imshow(image), plt.title('Original'), plt.axis('off')
    plt.subplot(5, 3, 2), plt.imshow(cropped_image), plt.title('Cropped'), plt.axis('off')
    plt.subplot(5, 3, 3), plt.imshow(normalized_image), plt.title('Normalized'), plt.axis('off')
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.subplot(5, 3, 4 + i), plt.imshow(normalized_image[:, :, i], cmap='gray'), plt.title(f'{channel}'), plt.axis('off')
    for i, channel in enumerate(['L', 'A', 'B']):
        plt.subplot(5, 3, 7 + i), plt.imshow(lab_image[:, :, i], cmap='gray'), plt.title(f'{channel}'), plt.axis('off')
    for i, channel in enumerate(['H', 'S', 'V']):
        plt.subplot(5, 3, 10 + i), plt.imshow(normalize_image(hsv_image[:, :, i]), cmap='gray'), plt.title(f'{channel}'), plt.axis('off')
    for i, channel in enumerate(['Y', 'Cb', 'Cr']):
        plt.subplot(5, 3, 13 + i), plt.imshow(ycbcr_image[:, :, i], cmap='gray'), plt.title(f'{channel}'), plt.axis('off')
    plt.tight_layout()
    plt.show()

image = io.read_image("input\\ingredients\\LPY\\LPY-P4_Vanillin-Original.jpg")
convert_and_show_image_in_various_color_space(image)