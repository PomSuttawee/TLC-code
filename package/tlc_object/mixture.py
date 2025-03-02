import logging
import numpy as np
from package.image_processing.general import crop
from package.image_processing.segmentation import threshold
from package.image_processing.data_extractor import general, rf

def validate_image(image: np.ndarray):
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be a numpy array")
    if image.size == 0:
        raise ValueError(f"Image cannot be empty")
    if image.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions")
    if image.ndim >= 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unexpected number of channels in image: {image.shape[2]}")

class Substance:
    def __init__(self, substance_index: int, peak_area: float, rf: float):
        self.substance_index = substance_index
        self.peak_area = peak_area
        self.rf = rf

class MixtureSingleColor:
    """
    Handles the segmentation and Rf value calculation for a single color channel.

    Attributes:
        channel_image (np.ndarray): Image array of the color channel.
        segmented_image (np.ndarray): Segmented image of the color channel.
        rf (List[float]): Calculated Rf values for the color channel.
    """
    def __init__(self, name: str, channel_image: np.ndarray):
        """
        Initializes the MixtureSingleColor with a given channel image.

        Args:
            channel_image (np.ndarray): Image array of the color channel.
        """
        log = logging.getLogger('mixture-color')
        log.info(f'Initializing color channel: {name}')
        
        # Validate inputs
        validate_image(channel_image)
        
        # Store basic attributes
        self.name = name
        self.channel_image = channel_image
        
        try:
            self._segment_image()
            self._calculate_rf_and_peak_area()
            self._process_substances()
            
        except Exception as e:
            log.error(f'Failed to process color channel {name}: {str(e)}')
            raise ValueError(f"Color channel processing failed: {str(e)}") from e
    
    def _segment_image(self):
        """Segments the image using thresholding."""
        self.segmented_image = threshold.segment_mixture(self.channel_image)
        
    def _calculate_rf_and_peak_area(self):
        """Calculates Rf values for the segmented image."""
        self.rf = rf.calculate_rf_detect_centroid(self.segmented_image)
        self.peak_area = general.calculate_peak_area(self.segmented_image)
    
    def _process_substances(self):
        """Processes the substances in the image."""
        self.substances = []
        for i, rf_value in enumerate(self.rf):
            self.substances.append(Substance(i, self.peak_area[i], rf_value))
    
    def get_image_with_centroids(self):
        return rf.get_image_with_centroids(self.segmented_image)
    
    @property
    def substance_count(self) -> int:
        """Return the number of substances detected in this channel"""
        return len(self.substances)
        
    @property 
    def has_substances(self) -> bool:
        """Return whether any substances were detected"""
        return len(self.substances) > 0

class Mixture:
    """
    Processes an image by segmenting it and calculating Rf values for different color channels.

    Attributes:
        name (str): Name of the mixture.
        original_image (np.ndarray): Original image data.
        detected_line_image (np.ndarray): Image with detected lines used for channel separation.
        red_channel_mixture (MixtureSingleColor): MixtureSingleColor for the red channel.
        green_channel_mixture (MixtureSingleColor): MixtureSingleColor for the green channel.
        blue_channel_mixture (MixtureSingleColor): MixtureSingleColor for the blue channel.
    """
    def __init__(self, name: str, image: np.ndarray):
        """
        Initializes the Mixture by detecting lines and splitting the image by color channels.

        Args:
            name (str): Name of the mixture.
            image (np.ndarray): Original image to be processed.
        """
        log = logging.getLogger('mixture')
        log.info(f'Initiate Mixture[{name}]')
        validate_image(image)
        
        self.name = name
        self.original_image = image
        
        try:
            self._preprocess_image()
            self._process_color_channel()
            
        except Exception as e:
            log.error(f'Failed to process ingredient {name}: {str(e)}')
            raise ValueError(f"Mixture processing failed: {str(e)}") from e
    
    def print_all_substances(self):
        """Prints all substances in the mixture."""
        for color_channel in [self.red_channel_mixture, self.green_channel_mixture, self.blue_channel_mixture]:
            print(f"Substances in {color_channel.name}:")
            for substance in color_channel.substances:
                print(f"Substance {substance.substance_index}: RF = {substance.rf}, Peak Area = {substance.peak_area}")
    
    def _preprocess_image(self):
        """Detects lines in the image and crops it accordingly."""
        self.line_detected_image = crop.detect_and_crop_houghlines(self.original_image)
    
    def _process_color_channel(self):
        """Creates MixtureSingleColor objects for each RGB channel."""
        log = logging.getLogger('mixture')
        
        # Process red channel
        log.debug(f'Initiate MixtureSingleColor: Red channel')
        self.red_channel_mixture = MixtureSingleColor(
            f"{self.name}_red",
            self.line_detected_image[:, :, 0]
        )
        
        # Process green channel
        log.debug(f'Initiate MixtureSingleColor: Green channel')
        self.green_channel_mixture = MixtureSingleColor(
            f"{self.name}_green",
            self.line_detected_image[:, :, 1]
        )
        
        # Process blue channel
        log.debug(f'Initiate MixtureSingleColor: Blue channel')
        self.blue_channel_mixture = MixtureSingleColor(
            f"{self.name}_blue",
            self.line_detected_image[:, :, 2]
        )
    
    def __str__(self):
        return f'Mixture[{self.name}]'