import numpy as np
import cv2

from package_2.image_processing.detect_paper import PaperDetector
from package_2.image_processing.detect_solvent_front_and_origin import TLCImageProcessor
from package_2.image_processing.segment_substance_spot import segment_hsv_threshold_range
from package_2.data_extractor.ingredient_data_extractor import extract_data

class Substance:
    def __init__(self, substance_index: int, image: np.ndarray, rf: float, slope: float, intercept: float, r_squared: float):
        self.substance_index = substance_index
        self.image = image
        self.rf = rf
        self.calibration_curve = (slope, intercept)
        self.r_squared = r_squared

class IngredientSingleChannel:
    def __init__(self, name: str, image: np.ndarray, concentration_list: list):
        self.name = name
        self.image = image
        self.concentration_list = concentration_list
        self.substances = self._create_substances()
    
    def _extract_data(self):
        return extract_data(self.image, self.concentration_list)
       
    def _create_substances(self):
        substance_data = self._extract_data()
        substances = {}
        for index, data in substance_data.items():
            substances[index] = Substance(index, data['image'], data['rf'], data['calibration_curve'][0], data['calibration_curve'][1], data['r_squared'])
        return substances
    
class Ingredient:
    def __init__(self, name: str, image: np.ndarray, concentration_list: list):
        self.name = name
        self.image = image
        self.concentration_list = concentration_list
        self.paper_image = self._process_image()
        self.segmented_image = self._segment_image()
        
        # self.gray_ingredient = IngredientSingleChannel(self.name + "_gray", self._convert_to_gray(self.segmented_image), self.concentration_list)
    
    def get_images(self):
        return [self.image, self.paper_image, self.segmented_image]
    
    def _process_image(self):
        return TLCImageProcessor.crop_solvent_front_and_origin(PaperDetector.crop_to_paper(self.image))
    
    def _segment_image(self):
        return segment_hsv_threshold_range(self.paper_image)

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)