from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image_path(input_type: str, substance_name: str = None) -> list:
    """
    Load image file paths from the specified directory.

    Args:
        input_type (str): Type of input (Mixtures/Ingredients)
        substnace_name (str): Name of the substance (only for Ingredients)

    Returns:
        list: List of image file paths.
    """
    if input_type == "mixtures":
        input_dir = os.path.join(os.getcwd(), 'input', 'mixtures')
    elif input_type == "ingredients" and substance_name:
        input_dir = os.path.join(os.getcwd(), 'input', 'ingredients', substance_name)
    else:
        raise ValueError("Invalid input_type or missing substance_name for ingredients")
    return [os.path.join(input_dir, image_name) for image_name in os.listdir(input_dir)]

def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from the specified file path in RGB format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_images(image_list: List[np.ndarray], name_list: List[str] = None) -> None:
    """
    Display a list of images in a grid layout.

    Args:
        image_list (list): List of images to display.
    """
    image_list = _flatten_image_list(image_list)
    num_columns = min(len(image_list), 4)
    num_rows = (len(image_list) + num_columns - 1) // num_columns
    plt.figure(figsize=(20, 5 * num_rows))
    for i, image in enumerate(image_list):
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(image if len(image.shape) == 3 else image, cmap='gray')
        if name_list:
            plt.title(name_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def _flatten_image_list(image_list: list) -> list[np.ndarray]:
    """
    Flatten a nested list of images into a single list.

    Args:
        image_list (list): Nested list of images.

    Returns:
        list[np.ndarray]: Flattened list of images.
    """
    result_images = list()
    for item in image_list:
        if isinstance(item, list):
            for sub_item in item:
                if hasattr(sub_item, 'shape'):
                    result_images.append(sub_item)
        elif hasattr(item, 'shape'):
            result_images.append(item)
    return result_images