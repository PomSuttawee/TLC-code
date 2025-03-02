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

class Lane:
    """Base class for lane objects within an ingredient image."""
    
    def __init__(self, lane_name: str, lane_image: np.ndarray):
        validate_image(lane_image, "lane_image")
        self.lane_name = lane_name
        self.lane_image = lane_image
    
    def __str__(self) -> str:
        """Base string representation of a lane."""
        return f"""
        Lane name: {self.lane_name}
        Lane image shape: {self.lane_image.shape}
        """

class VerticalLane(Lane):
    """
    Represents a vertical lane within an ingredient image.

    Attributes:
        lane_name (str): Identifies the lane.
        lane_image (np.ndarray): Image array of the vertical lane.
        rf (List[float]): RF values calculated from the lane in ascending order.
    """
    def __init__(self, lane_name: str, lane_image: np.ndarray):
        """
        Initializes the VerticalLane object and computes the RF values.

        Args:
            lane_name (str): Name of the lane.
            lane_image (np.ndarray): Image array of the vertical slice.
        """
        super().__init__(lane_name, lane_image)
        self.rf = rf.calculate_rf_detect_centroid(lane_image)
    
    def get_image_with_centroids(self) -> np.ndarray:
        """
        Returns the lane image with RF values annotated.

        Returns:
            np.ndarray: Image array with RF values annotated.
        """
        return rf.get_image_with_centroids(self.lane_image)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the lane, including shape and RF values.
        """
        return f"""
        Lane name: {self.lane_name}
        Lane image shape: {self.lane_image.shape}
        RF values: {self.rf}
        """

class HorizontalLane(Lane):
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
        super().__init__(lane_name, lane_image)
        validate_concentration(concentration)
        self.best_fit_line, self.r2 = calibration.calculate_best_fit_line_for_image(lane_image, concentration)

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
        
class Substance:
    def __init__(self, substance_index: int, vertical_lane: VerticalLane, horizontal_lane: HorizontalLane):
        self.substance_index = substance_index
        self.slope = horizontal_lane.best_fit_line[0]
        self.intercept = horizontal_lane.best_fit_line[1]
        self.rf = vertical_lane.rf[substance_index]
        self.mixture_group = None

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
        log.info(f'Initializing color channel: {name}')
        
        # Validate inputs
        validate_image(single_color_image, "single_color_image")
        validate_concentration(concentration)
        
        # Store basic attributes
        self.name = name
        self.single_color_image = single_color_image
        self.concentration = concentration
        
        try:
            # Process the image in stages
            self._segment_image()
            self._process_vertical_lanes()
            self._process_horizontal_lanes()
            self._process_substances()
            
        except Exception as e:
            log.error(f"Failed to process color channel {name}: {str(e)}")
            raise ValueError(f"Color channel processing failed: {str(e)}") from e

    def _segment_image(self):
        """Segments the image to identify lane structures."""
        log = logging.getLogger('ingredient-color')
        log.debug(f'Segmenting image')
        
        self.segmented_image = threshold.segment_ingredient(self.single_color_image)
        if self.segmented_image is None:
            raise ValueError("Segmentation failed, resulting in None")

    def _process_vertical_lanes(self):
        """Extracts and processes vertical lanes from the segmented image."""
        log = logging.getLogger('ingredient-color')
        log.debug(f'Cropping vertical lanes')
        
        self.vertical_lane_images = threshold.crop_vertically(self.segmented_image)
        if not self.vertical_lane_images:
            raise ValueError("No vertical lanes found")
        
        log.debug(f'Creating VerticalLane objects')
        self.vertical_lanes = [
            VerticalLane(f'V{i}', lane) 
            for i, lane in enumerate(self.vertical_lane_images)
        ]

    def _process_horizontal_lanes(self):
        """Extracts and processes horizontal lanes from the segmented image."""
        log = logging.getLogger('ingredient-color')
        log.debug(f'Cropping horizontal lanes')
        
        self.horizontal_lane_images = threshold.crop_horizontally(self.segmented_image)
        if not self.horizontal_lane_images:
            raise ValueError("No horizontal lanes found")
        
        log.debug(f'Creating HorizontalLane objects')
        self.horizontal_lanes = [
            HorizontalLane(f'H{i}', lane, self.concentration) 
            for i, lane in enumerate(self.horizontal_lane_images)
        ]
    
    def _process_substances(self):
        """Extracts and processes peaks from the vertical and horizontal lanes."""
        log = logging.getLogger('ingredient-color')
        log.debug(f'Creating Peak objects')
        
        self.substances = []
        for i, horizontal_lane in enumerate(self.horizontal_lanes):
            self.substances.append(Substance(i, self.vertical_lanes[-2], horizontal_lane))
    
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
            # Preprocess the image (cropping and line detection)
            self._preprocess_image()
            
            # Process individual color channels
            self._process_color_channels()
            
        except Exception as e:
            log.error(f"Failed to process ingredient {name}: {str(e)}")
            raise ValueError(f"Ingredient processing failed: {str(e)}") from e
    
    def _preprocess_image(self):
        """Handles image preprocessing steps: cropping and line detection."""
        log = logging.getLogger('ingredient')
        
        # Step 1: Crop image to largest contour
        log.debug(f'Cropping image to largest contour')
        self.cropped_image = crop.crop_to_largest_contour(self.original_image)
        if self.cropped_image is None:
            raise ValueError("Failed to crop image to largest contour")
        
        # Step 2: Detect and crop lines
        log.debug(f'Detecting lines in cropped image')
        self.line_detected_image = crop.detect_and_crop_houghlines(self.cropped_image)
        if self.line_detected_image is None:
            raise ValueError("Failed to detect lines in the image")
    
    def _process_color_channels(self):
        """Creates IngredientSingleColor objects for each RGB channel."""
        log = logging.getLogger('ingredient')
        
        # Process red channel
        log.debug(f'Initiate IngredientSingleColor: Red channel')
        self.red_channel_ingredient = IngredientSingleColor(
            f"{self.name}_red", 
            self.line_detected_image[:, :, 0], 
            self.concentration
        )
        
        # Process green channel
        log.debug(f'Initiate IngredientSingleColor: Green channel')
        self.green_channel_ingredient = IngredientSingleColor(
            f"{self.name}_green", 
            self.line_detected_image[:, :, 1], 
            self.concentration
        )
        
        # Process blue channel
        log.debug(f'Initiate IngredientSingleColor: Blue channel')
        self.blue_channel_ingredient = IngredientSingleColor(
            f"{self.name}_blue", 
            self.line_detected_image[:, :, 2], 
            self.concentration
        )

    def print_all_substances(self):
        for color in ['red', 'green', 'blue']:
            print(f'\n{color.upper()}')
            for substance in getattr(self, f'{color}_channel_ingredient').substances:
                print(f'Substance {substance.substance_index + 1}: RF {substance.rf}, Slope {substance.slope}, Intercept {substance.intercept}')

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