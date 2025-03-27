import numpy as np
import cv2

from package.image_processing.crop_paper import PaperDetector
from package.image_processing.crop_solvent_front_and_origin import TLCImageProcessor
from package.image_processing.crop_substance_spot import crop_substance_spot_mixture
from package.data_extractor.mixture_data_extractor import extract_data

class Substance():
    def __init__(self, substance_name: str, rf: float, peak_area: float):
        self.substance_name = substance_name
        self.rf = rf
        self.peak_area = peak_area
        print(f"Substance {self.substance_name} created.")
        print(f'Rf: {self.rf}, Peak Area: {self.peak_area}\n')

class MixtureSingleChannel():
    def __init__(self, name: str, image: np.ndarray, bounding_boxes: list):
        self.name = name
        self.image = image
        self.bounding_boxes = bounding_boxes
        self.substances = self._create_substances()
    
    def _extract_data(self):
        return extract_data(self.image, self.bounding_boxes)
    
    def _create_substances(self) -> dict:
        substance_data = self._extract_data()
        substances = {}
        for name, data in substance_data.items():
            substances[name] = Substance(name, data['rf'], data['peak_area'])    
        return substances
    
class Mixture():
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.paper_image = self._process_image()
        self.segmented_image, self.bounding_boxes = self._segment_image()
        
        # import os
        # path_paper = 'C:\\Users\\Suttawee\\Desktop\\TLC-code\\OUTPUT\\paper_image'
        # path_segment = 'C:\\Users\\Suttawee\\Desktop\\TLC-code\\OUTPUT\\segment_image'
        # cv2.imwrite(os.path.join(path_paper, f'{self.name[:-4]}_input.png'), self.paper_image)
        # cv2.imwrite(os.path.join(path_segment, f'{self.name[:-4]}_mask_algo.png'), self.segmented_image)
        
        self.gray_mixture = MixtureSingleChannel(self.name + "_gray", self._convert_to_gray(self.segmented_image), self.bounding_boxes)
        
    def get_images(self):
        return [self.image, self.paper_image, self.segmented_image]
    
    def _process_image(self):
        return TLCImageProcessor.crop_solvent_front_and_origin(PaperDetector.crop_to_paper(self.image))
    
    def _segment_image(self):
        return crop_substance_spot_mixture(self.paper_image)
    
    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)