import logging
from package.io import image_io
from package.tlc_object.ingredient import Ingredient
from package.tlc_object.mixture import Mixture
from package.tlc_object.tlc_analyzer import TLCAnalyzer

def initialize_ingredient(ingredient_image_path: str) -> Ingredient:
    if not ingredient_image_path:
        raise ValueError("No ingredient path found.")
    ingredient_name = ingredient_image_path.split('\\')[-1]
    ingredient_image = image_io.read_image(ingredient_image_path)
    ingredient_concentration = [5, 8.33, 16.67, 33.33, 50, 66.67, 83.33, 100]
    ingredient = Ingredient(ingredient_name, ingredient_image, ingredient_concentration)
    return ingredient

def initailize_mixture(mixture_image_path: str) -> Mixture:   
    if not mixture_image_path:
        raise ValueError("No mixture path found.")
    mixture_image = image_io.read_image(mixture_image_path)
    mixture = Mixture(name = mixture_image_path.split('\\')[-1], image = mixture_image)
    return mixture

def test_ingredient() -> None:
    ingredient_image_paths = image_io.load_image_path(input_type="ingredients")
    for path in ingredient_image_paths:
        ingredient = initialize_ingredient(path)
        logging.info(f"Initialized ingredient: {ingredient.name}")
        images = ingredient.get_images()
        image_io.display_images(images)

def test_mixture() -> None:
    mixture_image_paths = image_io.load_image_path(input_type="mixtures")
    for path in mixture_image_paths:
        mixture = initailize_mixture(path)
        logging.info(f"Initialized mixture: {mixture.name}")
        images = mixture.get_images()
        image_io.display_images(images)

def run_all_samples() -> None:
    ingredient_image_paths = image_io.load_image_path(input_type="ingredients")
    mixture_image_paths = image_io.load_image_path(input_type="mixtures")
    
    ing_dict = {
        '5CY': 0,
        'LPY': 2,
        'NGG': 6
    }
    
    mixture_ing_dict = {
        0: ['5CY', 'LPY', 'NGG'],
        1: ['5CY', 'LPY'],
        2: ['5CY', 'NGG'],
        3: ['LPY', 'NGG'],
        4: ['5CY', 'LPY'],
        5: ['5CY', 'NGG'],
        6: ['LPY', 'NGG'],
        7: ['5CY', 'LPY', 'NGG'],
        8: ['5CY', 'LPY', 'NGG'], 
    }
    
    for mixture_index, ingredient_indices in mixture_ing_dict.items():
        print(f"Mixture {mixture_index+1} with ingredients {ingredient_indices}")
        mixture = initailize_mixture(mixture_image_paths[mixture_index])
        ingredients = [initialize_ingredient(ingredient_image_paths[ing_dict[i]]) for i in ingredient_indices]
        
        tlc_analyzer = TLCAnalyzer(mixture, ingredients)
        tlc_analyzer.print_result()
    
    for mixture_index, ingredient_indices in mixture_ing_dict.items():
        new_mixture_index = 9 + mixture_index
        print(f"Mixture {new_mixture_index+1} with ingredients {ingredient_indices}")
        mixture = initailize_mixture(mixture_image_paths[new_mixture_index])
        ingredients = [initialize_ingredient(ingredient_image_paths[ing_dict[i]]) for i in ingredient_indices]
        
        tlc_analyzer = TLCAnalyzer(mixture, ingredients)
        tlc_analyzer.print_result()
    

def main() -> None:
    ingredient_image_paths = image_io.load_image_path(input_type="ingredients")
    mixture_image_paths = image_io.load_image_path(input_type="mixtures")
    
    MIXTURE_INDEX = 0
    mixture = initailize_mixture(mixture_image_paths[MIXTURE_INDEX])
    
    selected_path = [ingredient_image_paths[2], ingredient_image_paths[6], ingredient_image_paths[0]]
    ingredients = [initialize_ingredient(path) for path in selected_path]
    
    logging.info("TLC analysis started.")
    tlc_analyzer = TLCAnalyzer(mixture, ingredients)
    tlc_analyzer.get_result_json()
    # tlc_analyzer.print_result()
    logging.info("TLC analysis completed.")
    
    # ingredient_image = [ingredients[i].get_images() for i in range(len(ingredients))]
    # mixture_image = [mixture.get_images()]
    # image_io.display_images(ingredient_image + mixture_image)


if __name__ == '__main__':
    # logging.basicConfig(format='%(name)s -> %(funcName)s: %(message)s', level=logging.INFO)
    main()