"""
Module for detecting and cropping TLC plate images based on solvent front and origin lines.
"""
import cv2
import numpy as np
from typing import List, Tuple

# Type aliases
Line = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Image = np.ndarray

# Constants for line detection
MIN_LINE_LENGTH_FACTOR = 0.15  # Minimum line length as factor of image width
MAX_ANGLE_DEVIATION = 5  # Maximum deviation from horizontal in degrees
TOP_RANGE = (0.025, 0.1)  # Valid range for top lines (as percentage of image height)
BOTTOM_RANGE = (0.9, 0.975)  # Valid range for bottom lines (as percentage of image height)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


class TLCLineDetector:
    """Class for detecting horizontal lines in TLC images."""
    
    @staticmethod
    def detect_lines(image: Image) -> List[Line]:
        """
        Detect line segments in an image using LSD algorithm.
        
        Args:
            image: Input color image
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
            
        Raises:
            ValueError: If no lines are found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        enhanced = clahe.apply(gray)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lines, _, _, _ = lsd.detect(enhanced)

        if lines is None:
            raise ValueError("No lines found in the image.")
            
        return [line[0] for line in lines]
    
    @staticmethod
    def is_valid_line(image: Image, line: Line) -> bool:
        """
        Check if a line meets all validity criteria.
        
        Args:
            image: Reference image for dimension calculations
            line: Line to validate as (x1, y1, x2, y2)
            
        Returns:
            True if line is valid, False otherwise
        """
        return (TLCLineDetector._has_sufficient_length(image, line) and 
                TLCLineDetector._is_horizontal(line) and 
                TLCLineDetector._is_in_valid_position(image, line))
    
    @staticmethod
    def _has_sufficient_length(image: Image, line: Line) -> bool:
        """Check if line length exceeds the minimum threshold."""
        x1, y1, x2, y2 = line
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        min_valid_length = image.shape[1] * MIN_LINE_LENGTH_FACTOR
        return line_length > min_valid_length
    
    @staticmethod
    def _is_horizontal(line: Line) -> bool:
        """Check if line is approximately horizontal."""
        x1, y1, x2, y2 = line
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        return angle <= MAX_ANGLE_DEVIATION or angle >= (180 - MAX_ANGLE_DEVIATION)
    
    @staticmethod
    def _is_in_valid_position(image: Image, line: Line) -> bool:
        """Check if line is in valid position (top or bottom of image)."""
        x1, y1, x2, y2 = line
        height = image.shape[0]
        
        # Calculate valid ranges in pixels
        valid_top_min, valid_top_max = TOP_RANGE[0] * height, TOP_RANGE[1] * height
        valid_bottom_min, valid_bottom_max = BOTTOM_RANGE[0] * height, BOTTOM_RANGE[1] * height
        
        # Check if line is in top range
        is_valid_top = (valid_top_min <= y1 <= valid_top_max and 
                       valid_top_min <= y2 <= valid_top_max)
        
        # Check if line is in bottom range
        is_valid_bottom = (valid_bottom_min <= y1 <= valid_bottom_max and 
                          valid_bottom_min <= y2 <= valid_bottom_max)
        
        return is_valid_top or is_valid_bottom
    
    @staticmethod
    def filter_valid_lines(image: Image, lines: List[Line]) -> List[Line]:
        """
        Filter lines to keep only valid ones.
        
        Args:
            image: Reference image for dimension calculations
            lines: List of lines to filter
            
        Returns:
            List of valid lines
        """
        return [line for line in lines if TLCLineDetector.is_valid_line(image, line)]
    
    @staticmethod
    def separate_top_bottom_lines(image: Image, lines: List[Line]) -> Tuple[List[Line], List[Line]]:
        """
        Separate lines into top and bottom groups.
        
        Args:
            image: Input image
            lines: List of valid lines
            
        Returns:
            Tuple containing (top_lines, bottom_lines)
        """
        top_lines = []
        bottom_lines = []
        
        mid_point = image.shape[0] // 2
        for line in lines:
            x1, y1, x2, y2 = line
            if y1 < mid_point and y2 < mid_point:
                top_lines.append(line)
            elif y1 > mid_point and y2 > mid_point:
                bottom_lines.append(line)
                
        return top_lines, bottom_lines
    
    @staticmethod
    def calculate_average_y(lines: List[Line]) -> float:
        """
        Calculate the average Y-coordinate across all line endpoints.
        
        Args:
            lines: List of lines
            
        Returns:
            Average Y-coordinate
        """
        if not lines:
            return 0.0
            
        total_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            total_y += y1 + y2
            
        return total_y / (len(lines) * 2)


class TLCImageProcessor:
    """Class for processing TLC plate images."""
    
    @staticmethod
    def crop_solvent_front_and_origin(image: Image) -> Image:
        """
        Detect and crop image between solvent front and origin lines.
        
        Args:
            image: Input TLC plate image
            
        Returns:
            Cropped image containing only the area between solvent front and origin
            
        Raises:
            ValueError: If required lines cannot be detected
        """
        lines = TLCLineDetector.detect_lines(image)
        valid_lines = TLCLineDetector.filter_valid_lines(image, lines)
        
        if not valid_lines:
            raise ValueError("No valid lines found in the image.")
        
        top_lines, bottom_lines = TLCLineDetector.separate_top_bottom_lines(image, valid_lines)
        
        if not top_lines:
            raise ValueError("No solvent front (top) lines detected.")
        elif not bottom_lines:
            raise ValueError("No origin (bottom) lines detected.")
        
        average_top_y = TLCLineDetector.calculate_average_y(top_lines)
        average_bottom_y = TLCLineDetector.calculate_average_y(bottom_lines)
        
        return TLCImageProcessor._crop_by_y_coordinate(image, (average_top_y, average_bottom_y))
    
    @staticmethod
    def _crop_by_y_coordinate(image: Image, y_range: Tuple[float, float]) -> Image:
        """Crop image based on y-coordinate range."""
        return image[int(y_range[0]):int(y_range[1]), :]


class TLCVisualizer:
    """Class for visualizing TLC image processing steps."""
    
    @staticmethod
    def get_process_images(image: Image) -> List[Image]:
        """
        Generate a series of images showing the processing steps.
        
        Args:
            image: Input TLC plate image
            
        Returns:
            List of images showing different processing stages
        """
        image_list = []
        
        # Step 1: Detect all lines
        try:
            lines = TLCLineDetector.detect_lines(image)
            all_lines_image = TLCVisualizer._draw_lines(image.copy(), lines)
            image_list.append(all_lines_image)
        except ValueError:
            return [image]  # Return original if no lines found
        
        # Step 2: Filter valid lines
        valid_lines = TLCLineDetector.filter_valid_lines(image, lines)
        if not valid_lines:
            return image_list
            
        valid_lines_image = TLCVisualizer._draw_lines(image.copy(), valid_lines)
        image_list.append(valid_lines_image)
        
        # Step 3: Separate top and bottom lines
        top_lines, bottom_lines = TLCLineDetector.separate_top_bottom_lines(image, valid_lines)
        
        if top_lines:
            top_lines_image = TLCVisualizer._draw_lines(image.copy(), top_lines)
            image_list.append(top_lines_image)
        
        if bottom_lines:
            bottom_lines_image = TLCVisualizer._draw_lines(image.copy(), bottom_lines)
            image_list.append(bottom_lines_image)
        else:
            return image_list
            
        # Step 4: Draw average lines
        avg_top_y = TLCLineDetector.calculate_average_y(top_lines)
        avg_bottom_y = TLCLineDetector.calculate_average_y(bottom_lines)
        
        avg_lines_image = image.copy()
        width = image.shape[1]
        cv2.line(avg_lines_image, (0, int(avg_top_y)), (width, int(avg_top_y)), (0, 255, 0), 2)
        cv2.line(avg_lines_image, (0, int(avg_bottom_y)), (width, int(avg_bottom_y)), (0, 255, 0), 2)
        image_list.append(avg_lines_image)
        
        # Step 5: Add cropped image
        cropped_image = TLCImageProcessor._crop_by_y_coordinate(image, (avg_top_y, avg_bottom_y))
        image_list.append(cropped_image)
        
        return image_list
    
    @staticmethod
    def _draw_lines(image: Image, lines: List[Line], color: Tuple[int, int, int] = (0, 255, 0), 
                   thickness: int = 4) -> Image:
        """Draw lines on an image."""
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return image
    
    @staticmethod
    def visualize_process(image: Image) -> None:
        """
        Visualize TLC image processing steps using matplotlib.
        
        Args:
            image: Input TLC plate image
        """
        image_list = TLCVisualizer.get_process_images(image)
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, len(image_list) // 2, figsize=(20, 10))
        axs = axs.ravel()
        # Handle case when only one image is returned
        if len(image_list) == 1:
            axs.imshow(image_list[0])
            axs.axis('off')
        else:
            for i, img in enumerate(image_list):
                axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[i].axis('off')
                
        plt.tight_layout()
        plt.show()