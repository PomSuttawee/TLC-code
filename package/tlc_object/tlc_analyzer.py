import logging
from typing import List, Dict, Any
import sympy
from package.tlc_object.mixture import Mixture, MixtureSingleColor
from package.tlc_object.ingredient import Ingredient, IngredientSingleColor
from package.peak_alignment import basic_threshold_range, dynamic_time_warping
from pprint import pprint

"""
1. Assign group to each ingredient's Rf based on the mixture's Rf.
2. Calculate the percentage of each ingredient in the mixture.
"""

def show_equation_solver_process(equation_data):
    """Demonstrates the internal processing of EquationSolver with real data"""
    print("\n===== EquationSolver Internal Process =====\n")
    
    print("Step 1: Raw Equation Data Input")
    print("---------------------------------")
    pprint(equation_data)
    
    print("\nStep 2: Creating Symbols")
    print("-----------------------")
    # Get ingredient names from the first mixture peak (any would work)
    first_mixture_idx = list(equation_data.keys())[0]
    ingredient_names = [name for name in equation_data[first_mixture_idx].keys() 
                        if name != 'Peak Area']
    
    symbol_dict = {name: sympy.Symbol(f'concentration_{name}') for name in ingredient_names}
    print("Ingredient names:", ingredient_names)
    print("Symbol dictionary:")
    for name, symbol in symbol_dict.items():
        print(f"  {name} → {symbol}")
    
    print("\nStep 3: Building Equations")
    print("-------------------------")
    equations = []
    for mixture_index, data in equation_data.items():
        peak_area = data['Peak Area']
        print(f"\nMixture Peak {mixture_index} (Area: {peak_area}):")
        
        # Show the equation being built
        each_ingredient_eq = []
        equation_str_parts = []
        for ingredient_name, ingredient_data in data.items():
            if ingredient_name == 'Peak Area':
                continue
            
            slope = ingredient_data['Slope']
            intercept = ingredient_data['Intercept']
            symbol = symbol_dict[ingredient_name]
            
            each_ingredient_eq.append(slope * symbol + intercept)
            equation_str_parts.append(f"{slope} * [{ingredient_name}] + {intercept}")
        
        print(f"  Equation: {' + '.join(equation_str_parts)} = {peak_area}")
        equation = sympy.Eq(sum(each_ingredient_eq), peak_area)
        equations.append(equation)
        print(f"  Sympy form: {equation}")
    
    print("\nStep 4: System of Equations")
    print("--------------------------")
    for i, eq in enumerate(equations):
        print(f"Equation {i}: {eq}")
    
    print("\nStep 5: Solving the System")
    print("-------------------------")
    symbols = list(symbol_dict.values())
    print(f"Solving for: {symbols}")
    try:
        result = sympy.solve(equations, symbols)
        print("\nRaw solution:")
        pprint(result)
        
        print("\nStep 6: Converting to Percentages")
        print("--------------------------------")
        ingredient_percentages = {}
        
        if result:
            if isinstance(result, dict):
                print("Processing dictionary solution...")
                total = sum(float(val) for val in result.values())
                ingredient_percentages = {
                    name: (float(result[symbol]) / total) * 100 if total > 0 else 0
                    for name, symbol in symbol_dict.items()
                }
            elif isinstance(result, list):
                print("Processing list solution...")
                result_dict = {str(symbol): value for symbol, value in zip(symbols, result[0])}
                total = sum(float(val) for val in result_dict.values())
                ingredient_percentages = {
                    name: (float(result_dict[str(symbol)]) / total) * 100 if total > 0 else 0
                    for name, symbol in symbol_dict.items()
                }
                
            print("\nRaw concentration values:")
            if isinstance(result, dict):
                for name, symbol in symbol_dict.items():
                    print(f"  {name}: {float(result[symbol])}")
            else:
                for i, name in enumerate(ingredient_names):
                    print(f"  {name}: {float(result[0][i])}")
                
            print(f"\nTotal concentration sum: {total}")
            
            print("\nFinal percentage results:")
            for name, percentage in ingredient_percentages.items():
                print(f"  {name}: {percentage:.2f}%")
            
            print(f"\nPercentage sum: {sum(ingredient_percentages.values()):.2f}%")
        
    except Exception as e:
        print(f"Error solving equations: {str(e)}")
    
    return ingredient_percentages

class EquationSolver:
    """
    Solves the equation to calculate ingredient percentages in the mixture.
    
    This class solves the equation for each color channel and calculates the 
    percentage of each ingredient in the mixture.
    """
    def __init__(self, equation_data: Dict[int, Dict[str, Any]]):
        """
        Initialize the equation solver.
        
        Args:
            equation_data: Dict containing equation data for calculating ingredient percentages
        """
        self.logger = logging.getLogger('tlc-analyzer.equation-solver')
        self.equation_data = equation_data
        self.ingredient_percentages = {}
        
        self._solve_equations()
    
    def _solve_equations(self) -> None:
        """Solve the equations for each color channel."""
        self.logger.debug('Solving equations')
        
        # Create a dict of symbols for each ingredient's concentration
        ingredient_names = [name for name in self.equation_data[0].keys() if name != 'Peak Area']
        symbol_dict = {name: sympy.Symbol(f'concentration_{name}') for name in ingredient_names}
        
        # Create equations for each mixture peak
        equations = []
        for mixture_index, data in self.equation_data.items():
            self.logger.debug(f'Creating equation for mixture index {mixture_index}')
            equations.append(self._create_equation(data, symbol_dict))
        
        # Get symbols as a list for solving
        symbols = list(symbol_dict.values())
        
        # Solve the equations
        try:
            result = sympy.solve(equations, symbols)
            self.logger.debug(f'Equation solution: {result}')
            
            # Store the results in ingredient_percentages
            if result:
                if isinstance(result, dict):
                    # Handle dict result format
                    total = sum(float(val) for val in result.values())
                    self.ingredient_percentages = {
                        name: (float(result[symbol]) / total) * 100 if total > 0 else 0
                        for name, symbol in symbol_dict.items()
                    }
                elif isinstance(result, list):
                    # Handle list result format (for single solution)
                    result_dict = {str(symbol): value for symbol, value in zip(symbols, result[0])}
                    total = sum(float(val) for val in result_dict.values())
                    self.ingredient_percentages = {
                        name: (float(result_dict[str(symbol)]) / total) * 100 if total > 0 else 0
                        for name, symbol in symbol_dict.items()
                    }
        except Exception as e:
            self.logger.error(f"Error solving equations: {str(e)}")
            self.ingredient_percentages = {name: 0 for name in ingredient_names}
            
    def _create_equation(self, data: Dict[str, Any], symbol_dict: Dict[str, sympy.Symbol]) -> sympy.Eq:
        """
        Create equation for a mixture peak.
        
        Args:
            data: Dict containing equation data for a mixture peak
            symbol_dict: Dictionary mapping ingredient names to their symbols
        
        Returns:
            Equation for the mixture peak
        """
        peak_area = data['Peak Area']
        
        each_ingredient_eq = []
        for ingredient_name, ingredient_data in data.items():
            if ingredient_name == 'Peak Area':
                continue
            
            slope = ingredient_data['Slope']
            intercept = ingredient_data['Intercept']
            symbol = symbol_dict[ingredient_name]
            each_ingredient_eq.append(slope * symbol + intercept)
        
        return sympy.Eq(sum(each_ingredient_eq), peak_area)
    
    def get_ingredient_percentages(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Get the calculated ingredient percentages.
        
        Returns:
            Dict containing ingredient percentages for each color channel
        """
        return self.ingredient_percentages

    def show_process(self):
        """Display the step-by-step solution process"""
        return show_equation_solver_process(self.equation_data)

class SingleColorAnalyzer:
    """
    Analyzes relationship between mixture and ingredients for a single color channel.
    
    This class maps ingredient peaks to mixture peaks and prepares data for 
    calculating the percentage of each ingredient in the mixture.
    """
    def __init__(self, mixture_single_color: MixtureSingleColor, ingredients_single_color: List[IngredientSingleColor], 
                 threshold: float = 0.075):
        """
        Initialize the analyzer for a single color channel.
        
        Args:
            mixture_single_color: MixtureSingleColor object containing mixture data
            ingredients_single_color: List of IngredientSingleColor objects
            threshold: Threshold value for peak alignment (default: 0.075)
        """
        self.logger = logging.getLogger('tlc-analyzer.single-color')
        self.mixture_single_color = mixture_single_color
        self.ingredients_single_color = ingredients_single_color
        self.threshold = threshold
        self.equation_data = {}
        
        self._validate_inputs()
        self._map_ingredients_to_mixture()
        self._create_equation_data()
        self._solve_equations()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not isinstance(self.mixture_single_color, MixtureSingleColor):
            raise TypeError(f'Expected MixtureSingleColor object, got {type(self.mixture_single_color)}')
        
        if not isinstance(self.ingredients_single_color, list):
            raise TypeError(f'Expected list of IngredientSingleColor object, got {type(self.ingredients_single_color)}')
        
        for ingredient in self.ingredients_single_color:
            if not isinstance(ingredient, IngredientSingleColor):
                raise TypeError(f'Expected IngredientSingleColor object, got {type(ingredient)}')
    
    def _map_ingredients_to_mixture(self) -> None:
        """Map ingredients' substance indices to mixture's substance indices."""
        self.logger.debug("Mapping ingredient peaks to mixture peaks")
        self.ingredient_to_mixture_map = self._align_peaks()
        
        # Extract mixture peak areas
        self.mixture_peak_area = {
            substance.substance_index: substance.peak_area 
            for substance in self.mixture_single_color.substances
        }
        
        # Extract ingredient slopes and intercepts
        self.ingredient_slope_intercept = {
            ingredient.name: {
                substance.substance_index: (substance.slope, substance.intercept) 
                for substance in ingredient.substances
            } 
            for ingredient in self.ingredients_single_color
        }
        
        # Map ingredient slopes and intercepts to mixture indices
        self.mapped_ingredient_slope_intercept = {ingredient.name: {} for ingredient in self.ingredients_single_color}
        
        for ingredient in self.ingredients_single_color:
            for ingredient_index, mixture_index in self.ingredient_to_mixture_map.items():
                self.mapped_ingredient_slope_intercept[ingredient.name][mixture_index] = \
                    self.ingredient_slope_intercept[ingredient.name][ingredient_index]
    
    def _create_equation_data(self) -> None:
        """
        Create equation data from mixture's peak area and ingredient's slope and intercept.
        
        Result structure:
        {
            Mixture Index: {
                'Peak Area': peak_area, 
                Ingredient Name: {'Slope': slope, 'Intercept': intercept}
            }
        }
        """
        self.logger.debug("Creating equation data")
        self.equation_data = {}
        
        for substance in self.mixture_single_color.substances:
            mixture_index = substance.substance_index
            peak_area = self.mixture_peak_area[mixture_index]
            self.equation_data[mixture_index] = {'Peak Area': peak_area}
            
            for ingredient in self.ingredients_single_color:
                if mixture_index in self.mapped_ingredient_slope_intercept[ingredient.name]:
                    slope, intercept = self.mapped_ingredient_slope_intercept[ingredient.name][mixture_index]
                    self.equation_data[mixture_index][ingredient.name] = {
                        'Slope': slope, 
                        'Intercept': intercept
                    }
        
        self._log_equation_data()
    
    def _log_equation_data(self) -> None:
        """Log the equation data for debugging purposes."""
        for mixture_index, value in self.equation_data.items():
            self.logger.debug(f'Mixture Index: {mixture_index}')
            self.logger.debug(f'\tPeak Area: {value["Peak Area"]}')
            
            for ingredient_name, ingredient_data in value.items():
                if ingredient_name == 'Peak Area':
                    continue
                self.logger.debug(f'\tIngredient Name: {ingredient_name}')
                self.logger.debug(f'\t\tIngredient Slope and Intercept: {ingredient_data}')
    
    def _solve_equations(self) -> None:
        """Solve the equations for each color channel."""
        self.logger.debug('Solving equations')
        self.equation_solver = EquationSolver(self.equation_data)
    
    def _align_peaks(self) -> Dict[int, int]:
        """
        Align ingredient peaks with mixture peaks.
        
        Returns:
            Dict mapping ingredient substance indices to mixture substance indices
        """
        return basic_threshold_range.align_peak(
            self.mixture_single_color, 
            self.ingredients_single_color, 
            self.threshold
        )

    def align_using_dtw(self) -> Dict[int, int]:
        """
        Align peaks using dynamic time warping algorithm.
        
        Returns:
            Dict mapping ingredient substance indices to mixture substance indices
        """
        return dynamic_time_warping.align_peak(
            self.mixture_single_color, 
            self.ingredients_single_color
        )

    def get_equation_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the equation data.
        
        Returns:
            Dict containing equation data for calculating ingredient percentages
        """
        return self.equation_data


class TLCAnalyzer:
    """
    Analyzes TLC data to calculate ingredient percentages in a mixture.
    
    This class coordinates the analysis of all color channels (RGB) and 
    combines the results to determine the composition of the mixture.
    """
    def __init__(self, mixture_object: Mixture, ingredient_objects: List[Ingredient], 
                 alignment_threshold: float = 0.075):
        """
        Initialize the TLC analyzer.
        
        Args:
            mixture_object: Mixture object containing mixture data
            ingredient_objects: List of Ingredient objects
            alignment_threshold: Threshold value for peak alignment
        """
        self.logger = logging.getLogger('tlc-analyzer')
        self.logger.info('Initializing TLCAnalyzer')
        
        self.mixture_object = mixture_object
        self.ingredient_objects = ingredient_objects if isinstance(ingredient_objects, list) else [ingredient_objects]
        self.alignment_threshold = alignment_threshold
        
        self._validate_inputs()
        self._initialize_color_analyzers()
        
        self.logger.info('TLCAnalyzer initialization complete')
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not isinstance(self.mixture_object, Mixture):
            raise TypeError(f'Expected Mixture object, got {type(self.mixture_object)}')
        
        if not isinstance(self.ingredient_objects, list):
            raise TypeError(f'Expected list of Ingredient object, got {type(self.ingredient_objects)}')
        
        if not all(isinstance(ingredient, Ingredient) for ingredient in self.ingredient_objects):
            raise TypeError('All ingredients must be Ingredient objects')
    
    def _initialize_color_analyzers(self) -> None:
        """Initialize analyzers for each color channel."""
        self.logger.debug('Initializing SingleColorAnalyzer for red channel')
        self.red_analyzer = SingleColorAnalyzer(
            self.mixture_object.red_channel_mixture,
            [ingredient.red_channel_ingredient for ingredient in self.ingredient_objects],
            self.alignment_threshold
        )
        
        self.logger.debug('Initializing SingleColorAnalyzer for green channel')
        self.green_analyzer = SingleColorAnalyzer(
            self.mixture_object.green_channel_mixture,
            [ingredient.green_channel_ingredient for ingredient in self.ingredient_objects],
            self.alignment_threshold
        )
        
        self.logger.debug('Initializing SingleColorAnalyzer for blue channel')
        self.blue_analyzer = SingleColorAnalyzer(
            self.mixture_object.blue_channel_mixture,
            [ingredient.blue_channel_ingredient for ingredient in self.ingredient_objects],
            self.alignment_threshold
        )
    
    def get_all_equation_data(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Get equation data for all color channels.
        
        Returns:
            Dict with keys 'red', 'green', 'blue' containing equation data for each channel
        """
        return {
            'red': self.red_analyzer.get_equation_data(),
            'green': self.green_analyzer.get_equation_data(),
            'blue': self.blue_analyzer.get_equation_data()
        }