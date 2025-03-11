import numpy as np
import cv2

from package_2.image_processing.detect_paper import crop_to_paper
from package_2.image_processing.detect_solvent_front_and_origin import crop_solvent_front_and_origin, visualize_process
from package_2.image_processing.segment_substance_spot import segment_hsv_threshold_range
from package_2.image_processing.crop_vertical_lane import crop_vertical_lane_images
from package_2.image_processing.crop_horizontal_lane import crop_horizontal_lane_images

from package_2.data_extractor.rf import calculate_rf_from_image
from package_2.data_extractor.calibration_curve import calculate_calibration_curve

class Substance:
    def __init__(self, substance_index: int, rf: float, slope: float, intercept: float, r_squared: float):
        self.substance_index = substance_index
        self.rf = rf
        self.slope = slope
        self.intercept = intercept
        self.r_squared = r_squared

class IngredientSingleChannel:
    def __init__(self, name: str, image: np.ndarray, concentration_list: list):
        self.name = name
        self.image = image
        self.concentration_list = concentration_list
        self.substances = self._create_substances(self._calculate_rf(), self._calculate_calibration_curve())
        
    def _find_candidate_vertical_lane_image(self, vertical_lane_images: list):
        candidate_index = None
        max_pixel_count = 0
        for i, vertical_lane_image in enumerate(vertical_lane_images):
            new_image = vertical_lane_image.copy()
            new_image[new_image > 0] = 1
            pixel_count = np.sum(new_image)
            if pixel_count > max_pixel_count:
                candidate_index = i
                max_pixel_count = pixel_count
        return vertical_lane_images[candidate_index]
    
    def _calculate_rf(self):
        vertical_lane_images = crop_vertical_lane_images(self.image)
        candidate_vertical_lane_image = self._find_candidate_vertical_lane_image(vertical_lane_images)
        return calculate_rf_from_image(candidate_vertical_lane_image)
    
    def _calculate_calibration_curve(self):
        horizontal_lane_images = crop_horizontal_lane_images(self.image)
        best_fit_line_list = calculate_calibration_curve(horizontal_lane_images, self.concentration_list)
        return best_fit_line_list
            
    def _create_substances(self, rf_list: list, best_fit_line_list: list):
        substances = []
        for i, rf in enumerate(rf_list):
            substance = Substance(i, rf, best_fit_line_list[i][0], best_fit_line_list[i][1], best_fit_line_list[i][2])
            substances.append(substance)
        return substances
    
class Ingredient:
    def __init__(self, name: str, image: np.ndarray, concentration_list: list):
        self.name = name
        self.image = image
        self.concentration_list = concentration_list
        self.paper_image = self._process_image()
        self.segmented_image = self._segment_image()
        
        self.gray_ingredient = IngredientSingleChannel(self.name + "_gray", self._convert_to_gray(self.segmented_image), self.concentration_list)

    def visualize_process(self):
        from package_2.image_processing import detect_paper, detect_solvent_front_and_origin, segment_substance_spot, crop_vertical_lane, crop_horizontal_lane
        from package_2.data_extractor import rf
        detect_paper.visualize_process(self.image)
        detect_solvent_front_and_origin.visualize_process(crop_to_paper(self.image))
        segment_substance_spot.visualize_process(self.paper_image)
        crop_vertical_lane.visualize_process(self._convert_to_gray(self.segmented_image))
        crop_horizontal_lane.visualize_process(self._convert_to_gray(self.segmented_image))
        
        vertical_lanes = crop_vertical_lane_images(self._convert_to_gray(self.segmented_image))
        candidate_vertical_lane = self.gray_ingredient._find_candidate_vertical_lane_image(vertical_lanes)
        rf.visualize_process(candidate_vertical_lane)
    
    def _process_image(self):
        return crop_solvent_front_and_origin(crop_to_paper(self.image))
    
    def _segment_image(self):
        return segment_hsv_threshold_range(self.paper_image)

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)