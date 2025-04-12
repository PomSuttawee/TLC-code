import logging
import sympy as sp
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
        self.equations = self._create_equations()
        self.estimated_concentration = self._solve_equations()
        
    def print_result(self):
        # Print Equations
        # print("EQUATIONS".center(80))
        # print("-" * 80)
        # for mixture_substance_name, equation_data in self.equations.items():
        #     equation = equation_data['equation']
        #     r_squared = equation_data['R_squared']
        #     print(f"{mixture_substance_name}: {equation} (R²: {r_squared:.4f})")
        # print("-" * 80)
        # print("\n")
        
        # Print Estimated Concentration
        if len(self.estimated_concentration) == 0:
            logger.warning("No estimated concentration found")
            return
        print("ESTIMATED CONCENTRATION".center(80))
        print("-" * 80)
        for ingredient_name, concentration in self.estimated_concentration.items():
            if ingredient_name.endswith("_ratio"):
                print(f"{ingredient_name}: {concentration}")
        print("-" * 80)
        print("\n")
    
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
            original_header = "ORIGINAL NAME".ljust(20)
            rf_header = "RF".center(12)
            diff_header = "DIFFERENCE".center(12)
            slope_header = "SLOPE".center(12)
            intercept_header = "INTERCEPT".center(12)
            r2_header = "R²".center(12)
            
            print(f"{substance_header}{original_header}{rf_header}{diff_header}{slope_header}{intercept_header}{r2_header}")
            print(h_line)
            
            # Print aligned substance data - no change needed here
            for substance_name, substance in substances.items():
                substance_col = substance_name.ljust(20)
                original_col = substance['original_name'].ljust(20)
                rf_col = f"{substance['rf']:.4f}".center(12)
                diff_col = f"{substance['difference']:.4f}".center(12)
                slope_col = f"{substance['slope']:.4f}".center(12)
                intercept_col = f"{substance['intercept']:.4f}".center(12)
                r2_col = f"{substance['r_squared']:.4f}".center(12)
                
                print(f"{substance_col}{original_col}{rf_col}{diff_col}{slope_col}{intercept_col}{r2_col}")
            
            print(h_line)
            print("\n")
        
    def _extract_data(self) -> tuple:
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
    
    def _align_data(self) -> dict:
        # Use mixture rf as a reference
        mixture_rf = {substance_name: substance['rf'] for substance_name, substance in self.mixture_data.items()}
        sorted_mixture_rf = sorted(mixture_rf.items(), key=lambda x: x[1])
        mixture_rf = {name: rf for name, rf in sorted_mixture_rf}
        ingredient = self.ingredient_data
        # Align ingredient data to mixture data
        aligned_ingredient_data = {}
        
        for ing_name, ing_substances in ingredient.items():
            sorted_ing_substances = sorted(ing_substances.items(), key=lambda x: x[1]['rf'])
            sorted_ing_substances = {name: data for name, data in sorted_ing_substances}
            new_substances = {}
            for sub_name, sub_data in sorted_ing_substances.items():
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
                        'original_name': sub_name,
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
    
    def _create_equations(self) -> dict:
        # Setup variable
        variables = {}
        for ingredient_name, ingredient_substance in self.aligned_ingredient_data.items():
            variables[ingredient_name] = sp.symbols(f"{ingredient_name[:6]}", positive=True)
        
        equations = {}
        for mixture_substance_name, mixture_substance_data in self.mixture_data.items():
            ingredient_expression = []
            total_r_squared = 0
            for ingredient_name, ingredient_substance in self.aligned_ingredient_data.items():
                if mixture_substance_name in ingredient_substance:
                    ingredient_expression.append(ingredient_substance[mixture_substance_name]['slope'] * variables[ingredient_name] + ingredient_substance[mixture_substance_name]['intercept'])
                    total_r_squared += ingredient_substance[mixture_substance_name]['r_squared']
            if len(ingredient_expression) == 0:
                continue
            total_r_squared /= len(ingredient_expression)
            
            # Create the equation
            # Mixture Peak Area = (Slope_1 * X_1 + Intercept_1) + (Slope_2 * X_2 + Intercept_2) + ... + (Slope_n * X_n + Intercept_n)
            # where X_i is the concentration of the i-th ingredient
            peak_area = mixture_substance_data['peak_area']
            equation = sp.Eq(peak_area, sum(ingredient_expression))
            equations[mixture_substance_name] = {
                'equation': equation,
                'R_squared': total_r_squared,
            }
        return equations
    
    def _solve_equations(self):
        # self.show_data()
        
        # Select equations with highest R² value
        equations = sorted(self.equations.items(), key=lambda x: x[1]['R_squared'], reverse=True)
        # print("Equations sorted by R² value:")
        # print("-" * 80)
        # for eq in equations:
        #     print(f'{eq[0]}: {eq[1]["equation"]} (R²: {eq[1]["R_squared"]:.4f})')
        # print("-" * 80)
        
        selected_equations = equations[:len(self.ingredient_object_list)]
        # print("\nSelected equations:")
        # print("-" * 80)
        # for eq in selected_equations:
        #     print(f'{eq[0]}: {eq[1]["equation"]} (R²: {eq[1]["R_squared"]:.4f})')
        # print("-" * 80)
        
        selected_equations = [eq[1]['equation'] for eq in selected_equations]
        
        # Solve the equations
        solutions = sp.solve(selected_equations, dict=True)
        
        # If no solutions found or multiple solution sets, handle appropriately
        if not solutions:
            logger.warning("No solutions found for the system of equations")
            return {}
        
        # Use first solution (typically there's just one)
        solution = solutions[0]
        if len(solutions) > 1:
            logger.info("Multiple solutions found, using the first one")
            
        # Extract the concentration values
        concentration_values = {}
        for symbol, value in solution.items():
            # Extract the ingredient name from the symbol name (e.g., "Ingred_concentration")
            symbol_name = str(symbol)
            for ingredient in self.ingredient_object_list:
                if ingredient.name[:6] in symbol_name:
                    concentration_values[ingredient.name] = value
        
        # Calculate ratios
        if concentration_values:
            # Normalize to get ratios
            min_conc = min(concentration_values.values())
            ratio_values = {name: conc/min_conc for name, conc in concentration_values.items()}
            
            # Add ratios to the result
            for name, ratio in ratio_values.items():
                concentration_values[f"{name}_ratio"] = ratio
        
        return concentration_values