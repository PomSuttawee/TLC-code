import logging
from package.image_processing.general import io
from package.tlc_object.ingredient import Ingredient
from package.tlc_object.mixture import Mixture
from package.tlc_object.tlc_analyzer import TLCAnalyzer

def initialize_ingredient(ingredient_image_path: str) -> Ingredient:
    if not ingredient_image_path:
        raise ValueError("No ingredient path found.")
    ingredient_name = ingredient_image_path.split('\\')[-1]
    ingredient_image = io.read_image(ingredient_image_path)
    ingredient_concentration = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    # ingredient.print_all_substances()
    
    mixture_image_paths = io.load_image_path(input_type="mixtures")
    mixture = initailize_mixture(mixture_image_paths[0])
    # mixture.print_all_substances()

    # TLC Analyzer
    tlc_analyzer = TLCAnalyzer(mixture, ingredient)
    red_percentage = tlc_analyzer.red_analyzer.equation_solver.show_process()
    logging.info("TLC analysis completed.")

if __name__ == '__main__':
    logging.basicConfig(format='%(name)s -> %(funcName)s: %(message)s', level=logging.INFO)
    main()