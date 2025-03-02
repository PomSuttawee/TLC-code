from typing import List
from package.tlc_object.mixture import MixtureSingleColor
from package.tlc_object.ingredient import IngredientSingleColor

def align_peak(mixture: MixtureSingleColor, ingredient_list: List[IngredientSingleColor], threshold: float = 0.05):
    """
    Align ingredient's peak with the mixture's peak.
    """
    mixture_substance_rf = {}
    for substance in mixture.substances:
        mixture_substance_rf[substance.substance_index] = substance.rf
    
    ingredient_substance_rf = {}
    for ingredient in ingredient_list:
        for substance in ingredient.substances:
            ingredient_substance_rf[substance.substance_index] = substance.rf
    
    ingredient_mixture_substance_map = {}
    for mixture_index, mixture_rf in mixture_substance_rf.items():
        for ingredient_index, ingredient_rf in ingredient_substance_rf.items():
            if abs(mixture_rf - ingredient_rf) < threshold:
                ingredient_mixture_substance_map[ingredient_index] = mixture_index
                break

    return ingredient_mixture_substance_map
    