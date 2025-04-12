import numpy as np
import pandas as pd
import cv2

from package.image_processing.crop_paper import PaperDetector
from package.image_processing.crop_solvent_front_and_origin import TLCImageProcessor, TLCVisualizer
from package.image_processing.crop_substance_spot import crop_substance_spot_ingredient
from package.data_extractor.ingredient_data_extractor import extract_data

class Substance:
    def __init__(self, name: str, rf: float, slope: float, intercept: float, r_squared: float):
        self.name = name
        self.rf = rf
        self.slope = slope
        self.intercept = intercept
        self.r_squared = r_squared

class IngredientSingleChannel:
    def __init__(self, name: str, image: np.ndarray, bounding_boxes: list, concentration_list: list):
        self.name = name
        self.image = image
        self.bounding_boxes = bounding_boxes
        self.concentration_list = concentration_list
        self.substances = self._create_substances()
    
    def _extract_data(self):
        return extract_data(self.image, self.bounding_boxes, self.concentration_list)
       
    def _create_substances(self):
        substance_data = self._extract_data()
        substances = {}
        for name, data in substance_data.items():
            substances[name] = Substance(name      = name,
                                         rf        = data['rf'],
                                         slope     = data['calibration_curve'][0],
                                         intercept = data['calibration_curve'][1],
                                         r_squared = data['calibration_curve'][2])
        return substances
    
class Ingredient:
    def __init__(self, name: str, image: np.ndarray, concentration_list: list):
        self.name = name
        self.image = image
        self.concentration_list = concentration_list
        self.paper_image = self._process_image()
        self.segmented_image, self.bounding_boxes = self._segment_image()
        
        
        # import os
        # path_paper = 'C:\\Users\\Suttawee\\Desktop\\TLC-code\\OUTPUT\\paper_image'
        # path_segment = 'C:\\Users\\Suttawee\\Desktop\\TLC-code\\OUTPUT\\segment_image'
        # cv2.imwrite(os.path.join(path_paper, f'{self.name[:-4]}_input.png'), self.paper_image)
        # cv2.imwrite(os.path.join(path_segment, f'{self.name[:-4]}_mask_algo.png'), self.segmented_image)
        
        inv_gray_image = self._convert_to_gray(self.segmented_image, inverse = True)
        inv_gray_image[inv_gray_image == 255] = 0
        
        self.single_channel_ingredient = IngredientSingleChannel(name = self.name + "_gray", 
                                                                 image = inv_gray_image,
                                                                 bounding_boxes = self.bounding_boxes,
                                                                 concentration_list = self.concentration_list)

    def print_substance(self):
        data_df = pd.DataFrame(columns = ['Name', 'Rf', 'Slope', 'Intercept', 'R_squared'])
        for name, substance in self.single_channel_ingredient.substances.items():
            data_df = pd.concat([data_df, pd.DataFrame([[name, substance.rf, substance.slope, substance.intercept, substance.r_squared]], columns = data_df.columns)], ignore_index=True)
        data_df = data_df.set_index('Name')
        print(data_df)
    
    def get_images(self):
        return [self.image, self.paper_image, self.segmented_image, self.get_label_image()]
    
    def get_label_image(self):
        segmented_image_with_boxes = self.segmented_image.copy()
        for i, horizontal_boxes in enumerate(self.bounding_boxes):
            for j, box in enumerate(horizontal_boxes):
                x, y, w, h = box
                cv2.rectangle(segmented_image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(segmented_image_with_boxes, f"H{i+1}-V{j+1}", (x + 5, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return segmented_image_with_boxes
    
    def _process_image(self):
        return TLCImageProcessor.crop_solvent_front_and_origin(PaperDetector.crop_to_paper(self.image))
    
    def _segment_image(self ):
        return crop_substance_spot_ingredient(self.paper_image)
    
    def _convert_to_gray(self, image: np.ndarray, inverse: bool = False) -> np.ndarray:
        if inverse:
            return cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)