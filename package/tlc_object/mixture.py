import logging
from typing import Dict, List
import numpy as np
from package.image_processing.general import io, crop
from package.image_processing.segmentation import threshold
from package.image_processing.data_extractor import rf

class MixtureSingleColor:
    """
    Handles the segmentation and Rf value calculation for a single color channel.

    Attributes:
        channel_image (np.ndarray): Image array of the color channel.
        segmented_image (np.ndarray): Segmented image of the color channel.
        rf (List[float]): Calculated Rf values for the color channel.
    """
    def __init__(self, channel_image: np.ndarray):
        """
        Initializes the MixtureSingleColor with a given channel image.

        Args:
            channel_image (np.ndarray): Image array of the color channel.
        """
        try:
            log = logging.getLogger('mixture-color')
            self.channel_image = channel_image
            log.debug(f'Segmenting image')
            self.segmented_image = threshold.segment_mixture(self.channel_image)
            log.debug(f'Calculating Rf values')
            self.rf = rf.calculate_rf(self.segmented_image)
        except Exception as e:
            log.error(f'Error initializing MixtureSingleColor: {e}')
            raise
    
    def get_channel_image(self) -> np.ndarray:
        """
        Returns the image array of the color channel.

        Returns:
            np.ndarray: Image array of the color channel.
        """
        return self.channel_image
    
    def get_segmented_image(self) -> np.ndarray:
        """
        Returns the segmented image of the color channel.

        Returns:
            np.ndarray: Segmented image of the color channel.
        """
        return self.segmented_image
    
    def get_rf(self) -> List[float]:
        """
        Returns the Rf values for the color channel.

        Returns:
            List[float]: Calculated Rf values for the color channel.
        """
        return self.rf

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
    log = logging.getLogger('mixture')
    
    def __init__(self, name: str, image: np.ndarray):
        """
        Initializes the Mixture by detecting lines and splitting the image by color channels.

        Args:
            name (str): Name of the mixture.
            image (np.ndarray): Original image to be processed.
        """
        self.log.info(f'Initiate Mixture[{name}]')
        self.name = name
        self.original_image = image
        try:
            self.detected_line_image = crop.detect_and_crop_houghlines(image)
            
            self.log.debug(f'Initiate MixtureSingleColor: Red channel ')
            self.red_channel_mixture = MixtureSingleColor(self.detected_line_image[:, :, 0])
            
            self.log.debug(f'Initiate MixtureSingleColor: Green channel')
            self.green_channel_mixture = MixtureSingleColor(self.detected_line_image[:, :, 1])
            
            self.log.debug(f'Initiate MixtureSingleColor: Blue channel')
            self.blue_channel_mixture = MixtureSingleColor(self.detected_line_image[:, :, 2])
            
            self.log.info(f'Completely initiate Mixture[{name}]')
        except Exception as e:
            self.log.error(f'Error initializing Mixture[{name}]: {e}')
            raise
    
    def __str__(self):
        return f'Mixture[{self.name}]'
    
    def get_channel_image(self, color_channel: str) -> np.ndarray:
        """
        Returns the channel image for a given color.

        Args:
            color_channel (str): Color channel to retrieve.

        Returns:
            np.ndarray: Image array of the color channel.
        """
        channel_map = {
            'red': self.red_channel_mixture,
            'green': self.green_channel_mixture,
            'blue': self.blue_channel_mixture
        }
        if color_channel not in channel_map:
            raise ValueError(f"Invalid color channel: {color_channel}")
        return channel_map[color_channel].get_channel_image()
    
    def get_segmented_image(self, color_channel: str) -> np.ndarray:
        """
        Returns the segmented image for a given color.

        Args:
            color_channel (str): Color channel to retrieve.

        Returns:
            np.ndarray: Segmented image of the color channel.
        """
        channel_map = {
            'red': self.red_channel_mixture,
            'green': self.green_channel_mixture,
            'blue': self.blue_channel_mixture
        }
        if color_channel not in channel_map:
            raise ValueError(f"Invalid color channel: {color_channel}")
        return channel_map[color_channel].get_segmented_image()
    
    def get_rf(self, color_channel: str) -> List[float]:
        """
        Returns the Rf values for a given color.

        Args:
            color_channel (str): Color channel to retrieve.

        Returns:
            List[float]: Calculated Rf values for the color channel.
        """
        channel_map = {
            'red': self.red_channel_mixture,
            'green': self.green_channel_mixture,
            'blue': self.blue_channel_mixture
        }
        if color_channel not in channel_map:
            raise ValueError(f"Invalid color channel: {color_channel}")
        return channel_map[color_channel].get_rf()