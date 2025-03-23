import logging
from package_2.io import image_io
from package_2.tlc_object.ingredient import Ingredient
from package_2.tlc_object.mixture import Mixture
from package_2.tlc_object.tlc_analyzer import TLCAnalyzer

def initialize_ingredient(ingredient_image_path: str) -> Ingredient:
    if not ingredient_image_path:
        raise ValueError("No ingredient path found.")
    ingredient_name = ingredient_image_path.split('\\')[-1]
    ingredient_image = image_io.read_image(ingredient_image_path)
    ingredient_concentration = [5e-2, 8.33e-2, 16.67e-2, 33.33e-2, 50e-2, 66.67e-2, 83.33e-2, 100e-2]
    ingredient = Ingredient(ingredient_name, ingredient_image, ingredient_concentration)
    return ingredient

def initailize_mixture(mixture_image_path: str) -> Mixture:   
    if not mixture_image_path:
        raise ValueError("No mixture path found.")
    mixture_image = image_io.read_image(mixture_image_path)
    mixture = Mixture(name = mixture_image_path.split('\\')[-1], image = mixture_image)
    return mixture

def main() -> None:
    ingredient_image_paths = image_io.load_image_path(input_type="ingredients")
    for path in ingredient_image_paths:
        ingredient = initialize_ingredient(path)
        logging.info(f"Initialized ingredient: {ingredient.name}")
        # images = ingredient.get_images()
        # image_io.display_images(images)
    
    mixture_image_paths = image_io.load_image_path(input_type="mixtures")
    for path in mixture_image_paths:
        mixture = initailize_mixture(path)
        logging.info(f"Initialized mixture: {mixture.name}")
        # images = mixture.get_images()
        # image_io.display_images(images)

    # TLC Analyzer
    # tlc_analyzer = TLCAnalyzer(mixture, ingredient)
    logging.info("TLC analysis completed.")

if __name__ == '__main__':
    logging.basicConfig(format='%(name)s -> %(funcName)s: %(message)s', level=logging.INFO)
    main()