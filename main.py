import logging
from package.image_processing.general import io
# from package.tlc_object.ingredient import Ingredient
# from package.tlc_object.mixture import Mixture
from package.tlc_object.tlc_analyzer import TLCAnalyzer

from package_2.tlc_object.ingredient import Ingredient
from package_2.tlc_object.mixture import Mixture

def initialize_ingredient(ingredient_image_path: str) -> Ingredient:
    if not ingredient_image_path:
        raise ValueError("No ingredient path found.")
    ingredient_name = ingredient_image_path.split('\\')[-1]
    ingredient_image = io.read_image(ingredient_image_path)
    ingredient_concentration = [5e-2, 8.33e-2, 16.67e-2, 33.33e-2, 50e-2, 66.67e-2, 83.33e-2, 100e-2]
    ingredient = Ingredient(ingredient_name, ingredient_image, ingredient_concentration)
    return ingredient

def initailize_mixture(mixture_image_path: str) -> Mixture:   
    if not mixture_image_path:
        raise ValueError("No mixture path found.")
    mixture_image = io.read_image(mixture_image_path)
    mixture = Mixture(name = mixture_image_path.split('\\')[-1], image = mixture_image)
    return mixture

def main():
    SUBSTANCE_NAME = 'LPY'  # [5CY / LPY / NGG]
    
    ingredient_image_paths = io.load_image_path(input_type="ingredients", substance_name=SUBSTANCE_NAME)
    ingredient = initialize_ingredient(ingredient_image_paths[-1])
    ingredient.visualize_process()
    
    mixture_image_paths = io.load_image_path(input_type="mixtures")
    mixture = initailize_mixture(mixture_image_paths[0])
    mixture.visualize_process()

    # # TLC Analyzer
    # tlc_analyzer = TLCAnalyzer(mixture, ingredient)
    # red_percentage = tlc_analyzer.red_analyzer.equation_solver.show_process()
    # logging.info("TLC analysis completed.")

if __name__ == '__main__':
    logging.basicConfig(format='%(name)s -> %(funcName)s: %(message)s', level=logging.INFO)
    main()