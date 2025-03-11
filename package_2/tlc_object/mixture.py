import numpy as np
import cv2

from package_2.image_processing.detect_paper import crop_to_paper
from package_2.image_processing.detect_solvent_front_and_origin import crop_solvent_front_and_origin
from package_2.image_processing.segment_substance_spot import segment_hsv_threshold_range

from package_2.data_extractor.rf import calculate_rf_from_image

class Substance():
    def __init__(self, substance_index: int, rf: float):
        self.substance_index = substance_index
        self.rf = rf

class MixtureSingleChannel():
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.substances = self._create_substances(self._calculate_rf())
    
    def _calculate_rf(self):
        return calculate_rf_from_image(self.image)
    
    def _create_substances(self, rf_list: list):
        substances = []
        for i, rf in enumerate(rf_list):
            substance = Substance(i, rf)
            substances.append(substance)
        return substances
    
class Mixture():
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.processed_image = self._process_image()
        self.segmented_image = self._segment_image()

        self.gray_mixture = MixtureSingleChannel(self.name + '_gray', self._convert_to_gray(self.segmented_image))
    
    def visualize_process(self):
        from package_2.image_processing import detect_paper, detect_solvent_front_and_origin
        from package_2.data_extractor import rf
        detect_paper.visualize_process(self.image)
        detect_solvent_front_and_origin.visualize_process(crop_to_paper(self.image))
        
        rf.visualize_process(self._convert_to_gray(self.segmented_image))
    
    def _process_image(self):
        return crop_solvent_front_and_origin(crop_to_paper(self.image))
    
    def _segment_image(self):
        return segment_hsv_threshold_range(self.processed_image)
    
    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)