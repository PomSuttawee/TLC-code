import logging
from package.image_processing.general import io
from typing import List
import numpy as np
from package.image_processing.general import crop
from package.image_processing.segmentation import threshold
from package.image_processing.data_extractor import rf, calibration

def validate_image(image: np.ndarray, name: str):
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.size == 0:
        raise ValueError(f"{name} cannot be empty")

def validate_concentration(concentration: List[float]):
    if len(concentration) == 0:
        raise ValueError("concentration list cannot be empty")

class HorizontalLane:
    """
    Represents a horizontal lane within an ingredient image.

    Attributes:
        lane_name (str): Identifies the lane.
        lane_image (np.ndarray): Image array of the horizontal lane.
        best_fit_line (List[float]): Coefficients describing the best-fit calibration line.
        r2 (float): R² score indicating the fit quality.
    """
    def __init__(self, lane_name: str, lane_image: np.ndarray, concentration: List[float]):
        """
        Initializes the HorizontalLane object and calculates the best-fit line.

        Args:
            lane_name (str): Name of the lane.
            lane_image (np.ndarray): Image array of the horizontal slice.
            concentration (List[float]): List of concentration values for calibration.
        """
        validate_image(lane_image, "lane_image")
        validate_concentration(concentration)
        self.lane_name = lane_name
        self.lane_image = lane_image
        self.best_fit_line, self.r2 = calibration.calculate_best_fit_line(lane_image, concentration)

    def __str__(self) -> str:
        """
        Returns a string representation of the lane, including shape and model fit stats.
        """
        return f"""
        Lane name: {self.lane_name}
        Lane image shape: {self.lane_image.shape}
        Best fit line: {self.best_fit_line}
        R2: {self.r2}
        """
        
class VerticalLane:
    """
    Represents a vertical lane within an ingredient image.

    Attributes:
        lane_name (str): Identifies the lane.
        lane_image (np.ndarray): Image array of the vertical lane.
        raw_rf_values (List[float]): RF values calculated from the lane.
    """
    def __init__(self, lane_name: str, lane_image: np.ndarray):
        """
        Initializes the VerticalLane object and computes the RF values.

        Args:
            lane_name (str): Name of the lane.
            lane_image (np.ndarray): Image array of the vertical slice.
        """
        validate_image(lane_image, "lane_image")
        self.lane_name = lane_name
        self.lane_image = lane_image
        self.rf = rf.calculate_rf(lane_image)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the lane, including shape and RF values.
        """
        return f"""
        Lane name: {self.lane_name}
        Lane image shape: {self.lane_image.shape}
        RF values: {self.raw_rf_values}
        """

class IngredientSingleColor:
    """
    Processes a single color channel of an ingredient image.

    Attributes:
        name (str): Name of the ingredient color.
        single_color_image (np.ndarray): Single color channel image data.
        concentration (List[float]): List of concentration values for calibration.
        segmented_image (np.ndarray): Binary or refined mask of the lane structures.
        vertical_lane_images (List[np.ndarray]): List of vertical lane images.
        vertical_lanes (List[VerticalLane]): List of vertical lane objects.
        horizontal_lane_images (List[np.ndarray]): List of horizontal lane images.
        horizontal_lanes (List[HorizontalLane]): List of horizontal lane objects.
    """
    def __init__(self, name: str, single_color_image: np.ndarray, concentration: List[float]):
        """
        Initializes the IngredientSingleColor class by segmenting the image and extracting lanes.

        Args:
            name (str): Unique identifier for the color channel.
            single_color_image (np.ndarray): Single-channel image data.
            concentration (List[float]): List of concentration values for calibration.
        """
        log = logging.getLogger('ingredient-color')
        validate_image(single_color_image, "single_color_image")
        validate_concentration(concentration)
        self.name = name
        self.single_color_image = single_color_image
        self.concentration = concentration

        log.debug(f'Segmenting image')
        self.segmented_image = threshold.segment_ingredient(self.single_color_image)
        if self.segmented_image is None:
            raise ValueError("Segmentation failed, resulting in None")
        
        log.debug(f'Cropping vertical lanes')
        self.vertical_lane_images = threshold.crop_vertically(self.segmented_image)
        if not self.vertical_lane_images:
            raise ValueError("No vertical lanes found")
        
        log.debug(f'Initiate VerticalLane')
        self.vertical_lanes = [VerticalLane(f'lane_{i}', lane) for i, lane in enumerate(self.vertical_lane_images)]
        
        log.debug(f'Cropping horizontal lanes')
        self.horizontal_lane_images = threshold.crop_horizontally(self.segmented_image)
        if not self.horizontal_lane_images:
            raise ValueError("No horizontal lanes found")
        
        log.debug(f'Initiate HorizontalLane')
        self.horizontal_lanes = [HorizontalLane(f'lane_{i}', lane, self.concentration) for i, lane in enumerate(self.horizontal_lane_images)]
    
    def __str__(self) -> str:
        """
        Returns a string representation of the color processing status and lane counts.
        """
        return f"""
        Ingredient name: {self.name}
        Single color image shape: {self.single_color_image.shape}
        Segmented image shape: {self.segmented_image.shape}
        Number of vertical lanes: {len(self.vertical_lanes)}
        Number of horizontal lanes: {len(self.horizontal_lanes)}
        """

    def get_rf(self) -> List[List[float]]:
        """
        Returns the RF values for each vertical lane in the color channel.

        Returns:
            List[List[float]]: List of RF value arrays for each vertical lane.
        """
        return [lane.rf for lane in self.vertical_lanes]
    
    def get_best_fit_line(self) -> List[List[float]]:
        """
        Returns the best-fit line coefficients for each horizontal lane in the color channel.

        Returns:
            List[List[float]]: List of best-fit line coefficient arrays for each horizontal lane.
        """
        return [lane.best_fit_line for lane in self.horizontal_lanes]
    
    def get_r2(self) -> List[float]:
        """
        Returns the R² scores for each horizontal lane in the color channel.

        Returns:
            List[float]: List of R² scores for each horizontal lane.
        """
        return [lane.r2 for lane in self.horizontal_lanes]

class Ingredient:
    """
    Stores and processes an ingredient's image in multiple color channels.

    Attributes:
        name (str): Name of the ingredient.
        original_image (np.ndarray): Original image data.
        concentration (List[float]): List of concentration values for calibration.
        cropped_image (np.ndarray): Image cropped to its largest contour.
        line_detected_image (np.ndarray): Image with detected lines used for channel separation.
        red_channel_ingredient (IngredientSingleColor): IngredientSingleColor for the red channel.
        green_channel_ingredient (IngredientSingleColor): IngredientSingleColor for the green channel.
        blue_channel_ingredient (IngredientSingleColor): IngredientSingleColor for the blue channel.
    """
    def __init__(self, name: str, image: np.ndarray, concentration: List[float]):
        """
        Initializes the Ingredient by cropping and detecting lines, then splits it by color channels.

        Args:
            name (str): Name of the ingredient.
            image (np.ndarray): Original ingredient image to be processed.
            concentration (List[float]): List of concentration values for calibration.
        """
        log = logging.getLogger('ingredient')
        log.info(f'Initiate Ingredient: {name}')
        validate_image(image, "image")
        validate_concentration(concentration)
        self.name = name
        self.original_image = image
        self.concentration = concentration
        
        try:
            log.debug(f'Cropping image to largest contour')
            self.cropped_image = crop.crop_to_largest_contour(self.original_image)
            if self.cropped_image is None:
                raise ValueError("Cropping to largest contour failed, resulting in None")
            
            log.debug(f'Detecting lines in cropped image')
            self.line_detected_image = crop.detect_and_crop_houghlines(self.cropped_image)
            if self.line_detected_image is None:
                raise ValueError("Line detection failed, resulting in None")
            
            log.debug(f'Initiate IngredientSingleColor: Red channel')
            self.red_channel_ingredient = IngredientSingleColor(self.name, self.line_detected_image[:, :, 0], self.concentration)
            
            log.debug(f'Initiate IngredientSingleColor: Green channel')
            self.green_channel_ingredient = IngredientSingleColor(self.name, self.line_detected_image[:, :, 1], self.concentration)
            
            log.debug(f'Initiate IngredientSingleColor: Blue channel')
            self.blue_channel_ingredient = IngredientSingleColor(self.name, self.line_detected_image[:, :, 2], self.concentration)
            
            log.info(f'Completely initiate Ingredient[{name}]')
        except Exception as e:
            log.error(f'Error initializing Ingredient: {e}')
            raise

    def __str__(self) -> str:
        """
        Returns a string representation of the ingredient overview, including shapes of processed images.
        """
        return f"""
        Image name: {self.name}
        Original image shape: {self.original_image.shape}
        Red channel processed image shape: {self.red_channel_ingredient.segmented_image.shape}
        Green channel processed image shape: {self.green_channel_ingredient.segmented_image.shape}
        Blue channel processed image shape: {self.blue_channel_ingredient.segmented_image.shape}
        """
    
    def get_channel_image(self, color_channel: str) -> np.ndarray:
        """
        Returns the single-channel image for the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            np.ndarray: Single color channel image data.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        if color_channel == 'red':
            return self.red_channel_ingredient.single_color_image
        elif color_channel == 'green':
            return self.green_channel_ingredient.single_color_image
        elif color_channel == 'blue':
            return self.blue_channel_ingredient.single_color_image
        else:
            raise ValueError(f"Invalid color channel: {color_channel}")
    
    def get_segmented_image(self, color_channel: str) -> np.ndarray:
        """
        Returns the segmented image for the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            np.ndarray: Segmented binary or refined mask of the lane structures.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        if color_channel == 'red':
            return self.red_channel_ingredient.segmented_image
        elif color_channel == 'green':
            return self.green_channel_ingredient.segmented_image
        elif color_channel == 'blue':
            return self.blue_channel_ingredient.segmented_image
        else:
            raise ValueError(f"Invalid color channel: {color_channel}")
    
    def get_vertical_lane_images(self, color_channel: str) -> List[np.ndarray]:
        """
        Returns the list of vertical lane images for the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            List[np.ndarray]: List of vertical lane image arrays.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        if color_channel == 'red':
            return self.red_channel_ingredient.vertical_lane_images
        elif color_channel == 'green':
            return self.green_channel_ingredient.vertical_lane_images
        elif color_channel == 'blue':
            return self.blue_channel_ingredient.vertical_lane_images
        else:
            raise ValueError(f"Invalid color channel: {color_channel}")
    
    def get_horizontal_lane_images(self, color_channel: str) -> List[np.ndarray]:
        """
        Returns the list of horizontal lane images for the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            List[np.ndarray]: List of horizontal lane image arrays.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        if color_channel == 'red':
            return self.red_channel_ingredient.horizontal_lane_images
        elif color_channel == 'green':
            return self.green_channel_ingredient.horizontal_lane_images
        elif color_channel == 'blue':
            return self.blue_channel_ingredient.horizontal_lane_images
        else:
            raise ValueError(f"Invalid color channel: {color_channel}")
        
    def get_rf(self, color_channel: str) -> List[List[float]]:
        """
        Returns the list of RF values for each vertical lane in the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            List[List[float]]: List of RF value arrays for each vertical lane.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        channel_map = {
            'red': self.red_channel_ingredient,
            'green': self.green_channel_ingredient,
            'blue': self.blue_channel_ingredient
        }
        if color_channel not in channel_map:
            raise ValueError(f"Invalid color channel: {color_channel}")
        return channel_map[color_channel].get_rf()
        
    def get_best_fit_line(self, color_channel: str):
        """
        Returns the list of best-fit line coefficients for each horizontal lane 
        in the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            List[List[float]]: List of best-fit line coefficient arrays for each horizontal lane.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        channel_map = {
            'red': self.red_channel_ingredient,
            'green': self.green_channel_ingredient,
            'blue': self.blue_channel_ingredient
        }
        if color_channel not in channel_map:
            raise ValueError(f"Invalid color channel: {color_channel}")
        return channel_map[color_channel].get_best_fit_line()
        
    def get_r2(self, color_channel: str) -> List[float]:
        """
        Returns the list of R² scores for each horizontal lane in the specified color channel.

        Args:
            color_channel (str): The color channel to retrieve ('red', 'green', or 'blue').

        Returns:
            List[float]: List of R² scores for each horizontal lane.

        Raises:
            ValueError: If the given color channel is not valid.
        """
        channel_map = {
            'red': self.red_channel_ingredient,
            'green': self.green_channel_ingredient,
            'blue': self.blue_channel_ingredient
        }
        if color_channel not in channel_map:
            raise ValueError(f"Invalid color channel: {color_channel}")
        return channel_map[color_channel].get_r2()