import logging
from package.tlc_object.ingredient import Ingredient
from package.tlc_object.mixture import Mixture

# Set up logger
logger = logging.getLogger(__name__)

class TLCAnalyzer:
    def __init__(self, mixture: Mixture, ingredient_list: Ingredient) -> None:
        self.mixture_object = mixture
        self.ingredient_object_list = ingredient_list
        
        self.mixture_data, self.ingredient_data = self._extract_data()
        self.aligned_ingredient_data = self._align_data()
    
    def show_data(self) -> None:
        # Define a horizontal line
        h_line = "-" * 80
        
        print(h_line)
        print("MIXTURE DATA".center(80))
        print(h_line)
        
        # Define table header for mixture data
        substance_header = "SUBSTANCE".ljust(30)
        rf_header = "RF".center(15)
        peak_area_header = "PEAK AREA".center(15)
        
        print(f"{substance_header}{rf_header}{peak_area_header}")
        print(h_line)
        
        # Print mixture data
        for name, substance in self.mixture_data.items():
            substance_name = name.ljust(30)
            rf_value = f"{substance['rf']:.4f}".center(15)
            peak_area_value = f"{substance['peak_area']:.2f}".center(15)
            
            print(f"{substance_name}{rf_value}{peak_area_value}")
        
        print(h_line)
        print("\n")
        
        # Ingredient data
        print(h_line)
        print("INGREDIENT DATA".center(80))
        print(h_line)
        
        for ingredient_name, substances in self.ingredient_data.items():
            print(f"Ingredient: {ingredient_name}".center(80))
            print(h_line)
            
            # Define table header for ingredient data
            substance_header = "SUBSTANCE".ljust(20)
            rf_header = "RF".center(12)
            slope_header = "SLOPE".center(12)
            intercept_header = "INTERCEPT".center(12)
            r2_header = "R²".center(12)
            
            print(f"{substance_header}{rf_header}{slope_header}{intercept_header}{r2_header}")
            print(h_line)
            
            # Print substance data
            for substance_name, substance in substances.items():
                substance_col = substance_name.ljust(20)
                rf_col = f"{substance['rf']:.4f}".center(12)
                slope_col = f"{substance['slope']:.4f}".center(12)
                intercept_col = f"{substance['intercept']:.4f}".center(12)
                r2_col = f"{substance['r_squared']:.4f}".center(12)
                
                print(f"{substance_col}{rf_col}{slope_col}{intercept_col}{r2_col}")
            
            print(h_line)
            print("\n")
        
        # Aligned ingredient data
        print(h_line)
        print("ALIGNED INGREDIENT DATA".center(80))
        print(h_line)
        
        for ingredient_name, substances in self.aligned_ingredient_data.items():
            print(f"Ingredient: {ingredient_name}".center(80))
            print(h_line)
        
            # Define table header for aligned ingredient data - removed ingredient header
            substance_header = "SUBSTANCE".ljust(20)
            rf_header = "RF".center(12)
            diff_header = "DIFFERENCE".center(12)
            slope_header = "SLOPE".center(12)
            intercept_header = "INTERCEPT".center(12)
            r2_header = "R²".center(12)
            
            print(f"{substance_header}{rf_header}{diff_header}{slope_header}{intercept_header}{r2_header}")
            print(h_line)
            
            # Print aligned substance data - no change needed here
            for substance_name, substance in substances.items():
                substance_col = substance_name.ljust(20)
                rf_col = f"{substance['rf']:.4f}".center(12)
                diff_col = f"{substance['difference']:.4f}".center(12)
                slope_col = f"{substance['slope']:.4f}".center(12)
                intercept_col = f"{substance['intercept']:.4f}".center(12)
                r2_col = f"{substance['r_squared']:.4f}".center(12)
                
                print(f"{substance_col}{rf_col}{diff_col}{slope_col}{intercept_col}{r2_col}")
            
            print(h_line)
            print("\n")
        
    def _extract_data(self):
        mixture_single_channel = self.mixture_object.single_channel_mixture
        ingredient_single_channel_list = [ingredient.single_channel_ingredient for ingredient in self.ingredient_object_list]
        
        logger.info(f"Extracting data from mixture: {self.mixture_object.name}")
        mixture_data = {}
        for name, substance in mixture_single_channel.substances.items():
            logger.debug(f"Extracting data from substance: {name}")
            mixture_data[substance.name] = {'rf': substance.rf, 'peak_area': substance.peak_area}
        logger.info(f"Completely extracted mixture data.")
        
        # Add ingredient data extraction
        ingredient_data = {}
        for idx, ingredient in enumerate(self.ingredient_object_list):
            logger.info(f"Extracting data from ingredient: {ingredient.name}")
            ingredient_data[ingredient.name] = {}
            for name, substance in ingredient_single_channel_list[idx].substances.items():
                logger.debug(f"Extracting data from substance: {name}")
                ingredient_data[ingredient.name][substance.name] = {
                    'rf': substance.rf, 
                    'slope': substance.slope,
                    'intercept': substance.intercept,
                    'r_squared': substance.r_squared
                }
        logger.info(f"Completely extracted ingredient data.")
        
        return mixture_data, ingredient_data
    
    def _align_data(self) -> None:
        # Use mixture rf as a reference
        mixture_rf = {substance_name: substance['rf'] for substance_name, substance in self.mixture_data.items()}
        ingredient = self.ingredient_data
        # Align ingredient data to mixture data
        aligned_ingredient_data = {}
        
        for ing_name, ing_substances in ingredient.items():
            new_substances = {}
            for sub_name, sub_data in ing_substances.items():
                sub_rf, sub_slope, sub_intercept, sub_r2 = sub_data['rf'], sub_data['slope'], sub_data['intercept'], sub_data['r_squared']
                best_match_name = None
                best_diff = float('inf')

                for mix_name, mix_rf in mixture_rf.items():
                    diff = abs(sub_rf - mix_rf)
                    if diff < best_diff:
                        best_diff = diff
                        best_match_name = mix_name
                
                if best_diff <= 0.05:
                    new_substances[best_match_name] = {
                        'rf': sub_rf,
                        'difference': best_diff,
                        'slope': sub_slope,
                        'intercept': sub_intercept,
                        'r_squared': sub_r2
                    }
                else:
                    continue
            aligned_ingredient_data[ing_name] = new_substances
        return aligned_ingredient_data
    
    def analyze(self) -> None:
        pass