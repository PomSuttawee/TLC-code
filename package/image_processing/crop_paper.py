"""
Module for detecting and cropping to paper boundaries in images.
"""
import numpy as np
import cv2
from typing import List

# Type aliases
Image = np.ndarray
Contour = np.ndarray

# Constants
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA = 0


class PaperDetector:
    """Class for detecting and cropping to paper boundaries in images."""
    
    @staticmethod
    def crop_to_paper(image: Image) -> Image:
        """
        Detect paper boundaries in an image and crop to those boundaries.
        
        Args:
            image: Input color image
            
        Returns:
            Image cropped to paper boundaries
            
        Raises:
            ValueError: If no paper boundaries can be detected
        """
        contours = PaperDetector._get_contours(image)
        if not contours:
            raise ValueError("No paper boundaries detected in the image.")
            
        largest_contour = PaperDetector._get_largest_contour(contours)
        return PaperDetector._crop_image_using_contour(image, largest_contour)
    
    @staticmethod
    def _get_contours(image: Image) -> List[Contour]:
        """
        Detect contours in the image that could represent paper boundaries.
        
        Args:
            image: Input color image
            
        Returns:
            List of detected contours
        """
        # Input validation
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray_blur = cv2.GaussianBlur(
            gray, 
            GAUSSIAN_BLUR_KERNEL_SIZE, 
            GAUSSIAN_BLUR_SIGMA
        )
        
        # Apply Otsu's thresholding
        threshold = cv2.threshold(
            gray_blur, 
            0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        
        # Find contours
        contours, _ = cv2.findContours(
            threshold, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    @staticmethod
    def _get_largest_contour(contours: List[Contour]) -> Contour:
        """
        Select the largest contour by area.
        
        Args:
            contours: List of contours
            
        Returns:
            Largest contour
            
        Raises:
            ValueError: If contours list is empty
        """
        if not contours:
            raise ValueError("No contours provided")
            
        return max(contours, key=cv2.contourArea)
    
    @staticmethod
    def _crop_image_using_contour(image: Image, contour: Contour) -> Image:
        """
        Create mask for the contour and crop the image to its boundaries.
        
        Args:
            image: Input color image
            contour: Contour to crop to
            
        Returns:
            Cropped image
        """
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        masked_image = cv2.bitwise_and(image, mask)
        
        gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(gray_masked)
        
        cropped_image = masked_image[y:y+h, x:x+w]
        cropped_image = PaperDetector._fill_black_borders(cropped_image)
        return cropped_image
    
    @staticmethod
    def _fill_black_borders(image: Image) -> Image:
        """
        Fill in black borders of an image with estimated background color.
        
        Args:
            image: Input color image with black borders
            
        Returns:
            Image with black borders filled in
        """
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        black_mask = (gray == 0)
        if not np.any(black_mask):
            return result
        
        # Process each color channel separately
        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel]
            non_zero = channel_data[~black_mask]
            if len(non_zero) > 0:
                background_color = np.median(non_zero)
                result[:, :, channel][black_mask] = background_color
        
        return result