import numpy as np
import cv2

from package_2.image_processing.detect_paper import crop_to_paper
from package_2.image_processing.detect_solvent_front_and_origin import crop_solvent_front_and_origin
from package_2.image_processing.segment_substance_spot import segment_hsv_threshold_range
from package_2.data_extractor.mixture_data_extractor import extract_data

class Substance():
    def __init__(self, substance_index: int, image: np.ndarray, rf: float, peak_area: float):
        self.substance_index = substance_index
        self.image = image
        self.rf = rf
        self.peak_area = peak_area

class MixtureSingleChannel():
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.substances = self._create_substances()
    
    def show_substances_data(self):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        n_substances = len(self.substances)
        fig = plt.figure(figsize=(18, 18))
        
        # Create custom grid layout
        gs = GridSpec(n_substances, 3, figure=fig)
        
        # Create a subplot that spans all rows in first column
        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.imshow(self.image, cmap='gray')
        ax_main.set_title('Mixture Image')
        
        # Create individual subplots for each substance
        for substance in self.substances.values():
            i = substance.substance_index
            
            # Substance image in second column
            ax_img = fig.add_subplot(gs[i, 1])
            ax_img.imshow(substance.image, cmap='gray')
            
            # Substance text info in third column
            ax_text = fig.add_subplot(gs[i, 2])
            ax_text.axis('off')
            ax_text.text(0.1, 0.5, f"Substance: {substance.substance_index}\nRF: {substance.rf}\nPeak Area: {substance.peak_area}", 
                        fontsize=12, ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def _extract_data(self):
        return extract_data(self.image)
    
    def _create_substances(self) -> dict:
        substance_data = self._extract_data()
        substances = {}
        for index, data in substance_data.items():
            substances[index] = Substance(index, data['spot_image'], data['rf'], data['peak_area'])    
        return substances
    
class Mixture():
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.paper_image = self._process_image()
        self.segmented_image = self._segment_image()
        self.gray_mixture = MixtureSingleChannel(self.name + '_gray', self._convert_to_gray(self.segmented_image))
    
    def visualize_process(self):
        from package_2.image_processing import detect_paper, detect_solvent_front_and_origin, segment_substance_spot
        from package_2.data_extractor import rf, peak_area
        detect_paper.visualize_process(self.image)
        detect_solvent_front_and_origin.visualize_process(crop_to_paper(self.image))
        segment_substance_spot.visualize_process(self.paper_image)
        
        rf.visualize_process(self._convert_to_gray(self.segmented_image))
        peak_area.visualize_process(self._convert_to_gray(self.segmented_image))
    
    def _process_image(self):
        return crop_solvent_front_and_origin(crop_to_paper(self.image))
    
    def _segment_image(self):
        return segment_hsv_threshold_range(self.paper_image)
    
    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)