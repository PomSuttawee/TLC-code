import logging
from typing import List, Dict, Any, Union, Tuple, Optional
import sympy
from package.tlc_object.mixture import Mixture, MixtureSingleColor
from package.tlc_object.ingredient import Ingredient, IngredientSingleColor
from package.peak_alignment import basic_threshold_range
from pprint import pprint

DEFAULT_ALIGNMENT_THRESHOLD = 0.075
LOGGER_NAME_BASE = 'tlc-analyzer'

class EquationSolver:
    """
    Solves the equation system to calculate ingredient percentages in a mixture.
    
    This class processes equation data from mixture and ingredient peaks,
    constructs and solves a system of equations to determine the relative
    percentages of each ingredient in the mixture.
    """
    def __init__(self, equation_data: Dict[int, Dict[str, Any]]):
        """
        Initialize the equation solver.
        
        Args:
            equation_data: Dict containing equation data for calculating ingredient percentages
                Structure: {mixture_index: {'Peak Area': value, ingredient_name: {'Slope': value, 'Intercept': value, 'R_squared': value}}}
        """
        self.logger = logging.getLogger(f'{LOGGER_NAME_BASE}.equation-solver')
        self.equation_data = equation_data
        self.ingredient_percentages = {}
        
        self._solve_equations()
    
    def _get_ingredient_names(self) -> List[str]:
        """
        Extract ingredient names from equation data.
        
        Returns:
            List of ingredient names
        """
        # Get any mixture peak data (they all contain the same ingredients)
        if not self.equation_data:
            self.logger.warning("No equation data available")
            return []
            
        first_mixture_idx = list(self.equation_data.keys())[0]
        return [name for name in self.equation_data[first_mixture_idx].keys() 
                if name != 'Peak Area']
    
    def _create_symbol_dict(self, ingredient_names: List[str]) -> Dict[str, sympy.Symbol]:
        """
        Create a dictionary mapping ingredient names to symbolic variables.
        
        Args:
            ingredient_names: List of ingredient names
            
        Returns:
            Dictionary mapping ingredient names to their sympy symbols
        """
        return {name: sympy.Symbol(f'concentration_{name}') for name in ingredient_names}
   
    def _create_equation(self, data: Dict[str, Any], symbol_dict: Dict[str, sympy.Symbol]) -> Tuple[sympy.Eq, float]:
        """
        Create equation for a mixture peak.
        
        Args:
            data: Dict containing equation data for a mixture peak
            symbol_dict: Dictionary mapping ingredient names to their symbols
        
        Returns:
            Tuple of (equation for the mixture peak, average R-squared value)
        """
        peak_area = data['Peak Area']
        
        each_ingredient_eq = []
        total_r_squared = 0
        ingredient_count = 0
        
        for ingredient_name, ingredient_data in data.items():
            if ingredient_name == 'Peak Area':
                continue
                
            slope = ingredient_data['Slope']
            intercept = ingredient_data['Intercept']
            total_r_squared += ingredient_data['R_squared']
            ingredient_count += 1
            
            symbol = symbol_dict[ingredient_name]
            each_ingredient_eq.append(slope * symbol + intercept)
            
        average_r_squared = total_r_squared / ingredient_count if ingredient_count > 0 else 0
        equation = sympy.Eq(sum(each_ingredient_eq), peak_area)
        
        return (equation, average_r_squared)
    
    def _select_best_equations(self, equations_with_r2: List[Tuple[sympy.Eq, float]], 
                              num_required: int) -> List[sympy.Eq]:
        """
        Select the best equations based on R-squared values.
        
        Args:
            equations_with_r2: List of tuples (equation, R-squared value)
            num_required: Number of equations needed (typically equal to number of ingredients)
            
        Returns:
            List of selected equations
        """
        # Sort by R-squared values in descending order
        sorted_equations = sorted(equations_with_r2, key=lambda x: x[1], reverse=True)
        
        # Select the best equations
        return [eq for eq, _ in sorted_equations[:num_required]]
 
    def _solve_equations(self) -> None:
        """
        Solve the equations for each color channel and calculate ingredient percentages.
        """
        self.logger.debug('Solving equations for ingredient percentages')
        
        # Get ingredient names and create symbol dictionary
        ingredient_names = self._get_ingredient_names()
        if not ingredient_names:
            self.logger.warning("No ingredients found in equation data")
            self.ingredient_percentages = {}
            return
            
        symbol_dict = self._create_symbol_dict(ingredient_names)
        symbols = list(symbol_dict.values())
        
        # Create equations for each mixture peak
        equations_with_r2 = []
        for mixture_index, data in self.equation_data.items():
            self.logger.debug(f'Creating equation for mixture index {mixture_index}')
            equations_with_r2.append(self._create_equation(data, symbol_dict))
        
        try:
            # Select best equations based on R-squared values
            number_of_equations_needed = len(ingredient_names)
            selected_equations = self._select_best_equations(equations_with_r2, number_of_equations_needed)
            
            if len(selected_equations) < number_of_equations_needed:
                self.logger.warning(
                    f"Not enough equations ({len(selected_equations)}) to solve for {number_of_equations_needed} ingredients"
                )
                self.ingredient_percentages = {name: 0 for name in ingredient_names}
                return
                
            self.logger.debug(f"Selected {len(selected_equations)} equations with highest R-squared values")
            
            # Solve the equation system
            self.result = sympy.solve(selected_equations, symbols, dict=True)
            self.logger.info(f'Equation solution: {self.result}')
            self.logger.info('Equation solving complete')
              
        except Exception as e:
            self.logger.error(f"Error solving equations: {str(e)}", exc_info=True)
            self.ingredient_percentages = {name: 0 for name in ingredient_names}
    
    def get_ingredient_percentages(self) -> Dict[str, float]:
        """
        Get the calculated ingredient percentages.
        
        Returns:
            Dict mapping ingredient names to their calculated percentage in the mixture
        """
        return self.ingredient_percentages


class SingleColorAnalyzer:
    """
    Analyzes relationship between mixture and ingredients for a single color channel.
    
    This class maps ingredient peaks to mixture peaks and prepares data for 
    calculating the percentage of each ingredient in the mixture.
    """
    def __init__(self, mixture_single_color: MixtureSingleColor, 
                 ingredients_single_color: List[IngredientSingleColor], 
                 threshold: float = DEFAULT_ALIGNMENT_THRESHOLD):
        """
        Initialize the analyzer for a single color channel.
        
        Args:
            mixture_single_color: MixtureSingleColor object containing mixture data
            ingredients_single_color: List of IngredientSingleColor objects
            threshold: Threshold value for peak alignment
        """
        self.logger = logging.getLogger(f'{LOGGER_NAME_BASE}.single-color')
        self.mixture_single_color = mixture_single_color
        self.ingredients_single_color = ingredients_single_color
        self.threshold = threshold
        self.equation_data = {}
        self.equation_solver = None
        
        # Process the data
        self._validate_inputs()
        self._process_data()
        
    def _validate_inputs(self) -> None:
        """
        Validate input parameters.
        
        Raises:
            TypeError: If inputs are not of expected types
        """
        if not isinstance(self.mixture_single_color, MixtureSingleColor):
            raise TypeError(f'Expected MixtureSingleColor object, got {type(self.mixture_single_color)}')
        
        if not isinstance(self.ingredients_single_color, list):
            raise TypeError(f'Expected list of IngredientSingleColor objects, got {type(self.ingredients_single_color)}')
        
        for idx, ingredient in enumerate(self.ingredients_single_color):
            if not isinstance(ingredient, IngredientSingleColor):
                raise TypeError(f'Expected IngredientSingleColor object at index {idx}, got {type(ingredient)}')
                
        if self.ingredients_single_color and not self.mixture_single_color.substances:
            self.logger.warning("Mixture doesn't contain any substances")
            
        for ingredient in self.ingredients_single_color:
            if not ingredient.substances:
                self.logger.warning(f"Ingredient '{ingredient.name}' doesn't contain any substances")
    
    def _process_data(self) -> None:
        """
        Process the data by mapping peaks, creating equation data, and solving equations.
        """
        self._map_ingredients_to_mixture()
        self._create_equation_data()
        self._solve_equations()
        
    def _map_ingredients_to_mixture(self) -> None:
        """
        Map ingredients' substance indices to mixture's substance indices.
        """
        self.logger.debug("Mapping ingredient peaks to mixture peaks")
        
        # Align peaks between mixture and ingredients
        self.ingredient_to_mixture_map = self._align_peaks()
        
        # Extract mixture peak areas
        self.mixture_peak_area = {
            substance.substance_index: substance.peak_area 
            for substance in self.mixture_single_color.substances
        }
        
        # Extract ingredient calibration data
        self.ingredient_data = {}
        for ingredient in self.ingredients_single_color:
            self.ingredient_data[ingredient.name] = {}
            for substance in ingredient.substances:
                self.ingredient_data[ingredient.name][substance.substance_index] = (
                    substance.slope, 
                    substance.intercept, 
                    substance.r_squared
                )
        
        # Map ingredient data to mixture indices
        self.mapped_ingredient_data = {ingredient.name: {} for ingredient in self.ingredients_single_color}
        
        for ingredient in self.ingredients_single_color:
            for ingredient_index, mixture_index in self.ingredient_to_mixture_map.items():
                if ingredient_index in self.ingredient_data[ingredient.name]:
                    self.mapped_ingredient_data[ingredient.name][mixture_index] = self.ingredient_data[ingredient.name][ingredient_index]
    
    def _create_equation_data(self) -> None:
        """
        Create equation data from mixture's peak area and ingredient's calibration data.
        
        Result structure:
        {
            mixture_index: {
                'Peak Area': peak_area, 
                ingredient_name: {'Slope': slope, 'Intercept': intercept, 'R_squared': r_squared}
            }
        }
        """
        self.logger.debug("Creating equation data from peak mappings")
        self.equation_data = {}
        
        # For each mixture substance, collect data for equation building
        for substance in self.mixture_single_color.substances:
            mixture_index = substance.substance_index
            peak_area = self.mixture_peak_area[mixture_index]
            
            # Initialize with peak area
            self.equation_data[mixture_index] = {'Peak Area': peak_area}
            
            # Add ingredient data for this peak
            ingredient_count = 0
            for ingredient in self.ingredients_single_color:
                if mixture_index in self.mapped_ingredient_data[ingredient.name]:
                    slope, intercept, r_squared = self.mapped_ingredient_data[ingredient.name][mixture_index]
                    self.equation_data[mixture_index][ingredient.name] = {
                        'Slope': slope, 
                        'Intercept': intercept,
                        'R_squared': r_squared
                    }
                    ingredient_count += 1
            
            # Remove mixture peaks with no ingredient data
            if ingredient_count == 0:
                del self.equation_data[mixture_index]
                self.logger.debug(f"Removed mixture peak {mixture_index} with no ingredient data")
        
        self._log_equation_data()
    
    def _log_equation_data(self) -> None:
        """Log the equation data for debugging purposes."""
        self.logger.debug(f"Created equation data for {len(self.equation_data)} mixture peaks")
        
        for mixture_index, data in self.equation_data.items():
            ingredients = [name for name in data.keys() if name != 'Peak Area']
            self.logger.debug(f"  Mixture peak {mixture_index}: Peak Area={data['Peak Area']:.4f}, "
                             f"Ingredients={', '.join(ingredients)}")
    
    def _solve_equations(self) -> None:
        """
        Solve the equations to determine ingredient percentages.
        """
        self.logger.debug('Solving equations using EquationSolver')
        if self.equation_data:
            self.equation_solver = EquationSolver(self.equation_data)
        else:
            self.logger.warning("No equation data available to solve")
    
    def _align_peaks(self) -> Dict[int, int]:
        """
        Align ingredient peaks with mixture peaks using threshold method.
        
        Returns:
            Dict mapping ingredient substance indices to mixture substance indices
        """
        return basic_threshold_range.align_peak(
            self.mixture_single_color, 
            self.ingredients_single_color, 
            self.threshold
        )

    def get_equation_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the equation data.
        
        Returns:
            Dict containing equation data for calculating ingredient percentages
        """
        return self.equation_data
        
    def get_ingredient_percentages(self) -> Dict[str, float]:
        """
        Get the calculated ingredient percentages.
        
        Returns:
            Dict mapping ingredient names to their percentage
        """
        if self.equation_solver:
            return self.equation_solver.get_ingredient_percentages()
        return {}


class TLCAnalyzer:
    """
    Analyzes TLC data to calculate ingredient percentages in a mixture.
    
    This class coordinates the analysis of all color channels (RGB) and 
    combines the results to determine the composition of the mixture.
    """
    def __init__(self, mixture_object: Mixture, ingredient_objects: List[Ingredient], 
                 alignment_threshold: float = DEFAULT_ALIGNMENT_THRESHOLD):
        """
        Initialize the TLC analyzer.
        
        Args:
            mixture_object: Mixture object containing mixture data
            ingredient_objects: List of Ingredient objects
            alignment_threshold: Threshold value for peak alignment
        """
        self.logger = logging.getLogger(LOGGER_NAME_BASE)
        self.logger.info('Initializing TLCAnalyzer')
        
        self.mixture_object = mixture_object
        self.ingredient_objects = ingredient_objects if isinstance(ingredient_objects, list) else [ingredient_objects]
        self.alignment_threshold = alignment_threshold
        
        # Color-specific analyzers
        self.red_analyzer = None
        self.green_analyzer = None
        self.blue_analyzer = None
        
        # Initialize and process
        self._validate_inputs()
        self._initialize_color_analyzers()
        
        self.logger.info('TLCAnalyzer initialization complete')
    
    def _validate_inputs(self) -> None:
        """
        Validate input parameters.
        
        Raises:
            TypeError: If inputs are not of expected types
            ValueError: If no ingredients are provided
        """
        if not isinstance(self.mixture_object, Mixture):
            raise TypeError(f'Expected Mixture object, got {type(self.mixture_object)}')
        
        if not self.ingredient_objects:
            raise ValueError('At least one ingredient must be provided')
        
        if not isinstance(self.ingredient_objects, list):
            raise TypeError(f'Expected list of Ingredient objects, got {type(self.ingredient_objects)}')
        
        for idx, ingredient in enumerate(self.ingredient_objects):
            if not isinstance(ingredient, Ingredient):
                raise TypeError(f'Ingredient at index {idx} must be an Ingredient object, got {type(ingredient)}')
    
    def _initialize_color_analyzers(self) -> None:
        """
        Initialize analyzers for each color channel (RGB).
        """
        # Initialize with logging for tracking analysis progress
        self.logger.info(f'Initializing color channel analyzers (threshold={self.alignment_threshold:.4f})')
        
        # Red channel
        self.logger.debug('Initializing SingleColorAnalyzer for red channel')
        self.red_analyzer = SingleColorAnalyzer(
            self.mixture_object.red_channel_mixture,
            [ingredient.red_channel_ingredient for ingredient in self.ingredient_objects],
            self.alignment_threshold
        )
        
        # Green channel
        self.logger.debug('Initializing SingleColorAnalyzer for green channel')
        self.green_analyzer = SingleColorAnalyzer(
            self.mixture_object.green_channel_mixture,
            [ingredient.green_channel_ingredient for ingredient in self.ingredient_objects],
            self.alignment_threshold
        )
        
        # Blue channel
        self.logger.debug('Initializing SingleColorAnalyzer for blue channel')
        self.blue_analyzer = SingleColorAnalyzer(
            self.mixture_object.blue_channel_mixture,
            [ingredient.blue_channel_ingredient for ingredient in self.ingredient_objects],
            self.alignment_threshold
        )
        
        self.logger.debug('All color channel analyzers initialized')
    
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
        
    def get_all_ingredient_percentages(self) -> Dict[str, Dict[str, float]]:
        """
        Get calculated ingredient percentages for all channels.
        
        Returns:
            Dict with keys 'red', 'green', 'blue' mapping to ingredient percentage results
        """
        return {
            'red': self.red_analyzer.get_ingredient_percentages(),
            'green': self.green_analyzer.get_ingredient_percentages(),
            'blue': self.blue_analyzer.get_ingredient_percentages()
        }