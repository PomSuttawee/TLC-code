import logging
from typing import List
from package.tlc_object.mixture import Mixture, MixtureColor
from package.tlc_object.ingredient import Ingredient, IngredientColor

class SingleColorAnalyzer:
    def __init__(self, mixture_single_color: MixtureColor, ingredients_single_color: List[IngredientColor]):
        self.mixture_single_color = mixture_single_color
        self.ingredient_single_color = ingredients_single_color
        self.ingredient_rf_group = self._assign_group()
        
    def _assign_group(self, threshold: float = 0.02):
        """
        Assign group to each ingredient's Rf based on the mixture's Rf.
        """
        mixture_rf_group = {group_num : self.mixture_single_color.raw_rf_values[group_num-1] for group_num in range(1, len(self.mixture_single_color.raw_rf_values) + 1)}
        
        ingredient_rf = dict()
        for ingredient in self.ingredient_single_color:
            ingredient_rf[ingredient.name] = ingredient.vertical_lanes[-2].raw_rf_values
        print(f'Mixture RF group: {mixture_rf_group}')
        print(f'Ingredient Rf: {ingredient_rf}')
        
class TLCAnalyzer:
    def __init__(self, mixture_object: Mixture, ingredient_objects: List[Ingredient]):
        log = logging.getLogger('tlc-analyzer')
        log.info(f'Initiate TLCAnalyzer')
        self.mixture_object = mixture_object
        self.ingredient_objects = ingredient_objects
        
        log.debug(f'Initiate SingleColorAnalyzer for red channel')
        self.red_analyzer = SingleColorAnalyzer(mixture_object.red_channel_mixture, [ingredient.red_channel_ingredient for ingredient in ingredient_objects])
        
        log.debug(f'Initiate SingleColorAnalyzer for green channel')
        self.green_analyzer = SingleColorAnalyzer(mixture_object.green_channel_mixture, [ingredient.green_channel_ingredient for ingredient in ingredient_objects])
        
        log.debug(f'Initiate SingleColorAnalyzer for blue channel')
        self.blue_analyzer = SingleColorAnalyzer(mixture_object.blue_channel_mixture, [ingredient.blue_channel_ingredient for ingredient in ingredient_objects])
        
        log.info(f'Completely initiate TLCAnalyzer') 
    """
    1. Assign group to each ingredient's Rf based on the mixture's Rf.
    2. Calculate the percentage of each ingredient in the mixture.
    """