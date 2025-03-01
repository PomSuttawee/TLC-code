import logging
import numpy as np
from package.image_processing.general import crop
from package.image_processing.segmentation import threshold
from package.image_processing.data_extractor import peak_area, rf
from package.image_processing.data_extractor.prak_area import calculate_peak_area_from_image

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
            
            self.rf = rf.calculate_rf_detect_centroid(self.segmented_image)
            self.peak_area = calculate_peak_area_from_image(self.segmented_image)
        except Exception as e:
            log.error(f'Error initializing MixtureSingleColor: {e}')
            raise
    
    def get_image_with_centroids(self):
        return rf.get_image_with_centroids(self.segmented_image)

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
            self.line_detected_image = crop.detect_and_crop_houghlines(image)
            
            self.log.debug(f'Initiate MixtureSingleColor: Red channel ')
            self.red_channel_mixture = MixtureSingleColor(self.line_detected_image[:, :, 0])
            
            self.log.debug(f'Initiate MixtureSingleColor: Green channel')
            self.green_channel_mixture = MixtureSingleColor(self.line_detected_image[:, :, 1])
            
            self.log.debug(f'Initiate MixtureSingleColor: Blue channel')
            self.blue_channel_mixture = MixtureSingleColor(self.line_detected_image[:, :, 2])
            
            self.log.info(f'Completely initiate Mixture[{name}]')
        except Exception as e:
            self.log.error(f'Error initializing Mixture[{name}]: {e}')
            raise
    
    def __str__(self):
        return f'Mixture[{self.name}]'