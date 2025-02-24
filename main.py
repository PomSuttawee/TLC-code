import logging
from package.image_processing.general import io
from package.tlc_object.ingredient import Ingredient
from package.tlc_object.mixture import Mixture
from package.tlc_object.tlc_analyzer import TLCAnalyzer

def initialize_ingredient(ingredient_image_path: str) -> list[Ingredient]:
    if not ingredient_image_path:
        raise ValueError("No ingredient path found.")
    ingredient_name = ingredient_image_path.split('\\')[-1]
    ingredient_image = io.read_image(ingredient_image_path)
    ingredient_concentration = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ingredient = Ingredient(ingredient_name, ingredient_image, ingredient_concentration)
    return ingredient

def show_ingredient_data(ingredient: Ingredient) -> None:
    for color in ['red', 'green', 'blue']:
        # Retrieve data
        channel_image = ingredient.get_channel_image(color)
        segmented_image = ingredient.get_segmented_image(color)
        vertical_lane_images = ingredient.get_vertical_lane_images(color)
        horizontal_lane_images = ingredient.get_horizontal_lane_images(color)
        rf = ingredient.get_rf(color)
        best_fit_line = ingredient.get_best_fit_line(color)
        r2 = ingredient.get_r2(color)
        
        # Show data
        print(f'\n{color}'.upper())
        for i, rf_value in enumerate(rf):
            print(f'{color.upper()} Vertical Lane {i + 1} Rf: {rf_value}')
        for i, best_fit_line in enumerate(best_fit_line):
            print(f'{color.upper()} Horizontal Lane {i + 1}:')
            print(f'    Slope: {best_fit_line[0]}')
            print(f'    Intercept: {best_fit_line[1]}')
            print(f'    R2: {r2[i]}')
        images = [channel_image, segmented_image] + vertical_lane_images + horizontal_lane_images
        names = [f'{color.upper()} Channel', f'{color.upper()} Segmented'] + [f'{color.upper()} Vertical Lane {i}' for i in range(1, len(vertical_lane_images) + 1)] + [f'{color.upper()} Horizontal Lane {i}' for i in range(1, len(horizontal_lane_images) + 1)]
        io.show_images(images, names)

def initailize_mixture(mixture_image_path: str) -> Mixture:   
    if not mixture_image_path:
        raise ValueError("No mixture path found.")
    mixture_image = io.read_image(mixture_image_path)
    mixture = Mixture(name = mixture_image_path.split('\\')[-1], image = mixture_image)
    return mixture

def show_mixture_data(mixture: Mixture) -> None:
    for color in ['red', 'green', 'blue']:
        # Retrieve data
        channel_image = mixture.get_channel_image(color)
        segmented_image = mixture.get_segmented_image(color)
        rf = mixture.get_rf(color)
        
        # Show data
        print(f'\n{color.upper()}')
        for i, rf_value in enumerate(rf):
            print(f'Vertical Lane {i + 1} Rf: {rf_value}')
        images = [channel_image, segmented_image]
        names = [f'{color.upper()} Channel', f'{color.upper()} Segmented']
        io.show_images(images, names)

def main():
    SUBSTANCE_NAME = 'LPY'  # [5CY / LPY / NGG]
    
    # Ingredients
    ingredient_image_paths = io.load_image_path(input_type="ingredients", substance_name=SUBSTANCE_NAME)
    ingredient = initialize_ingredient(ingredient_image_paths[-1])
    show_ingredient_data(ingredient)
    
    # Mixture
    mixture_image_paths = io.load_image_path(input_type="mixtures")
    mixture = initailize_mixture(mixture_image_paths[-1])
    show_mixture_data(mixture)

    # # TLC Analyzer
    # tlc_analyzer = TLCAnalyzer(mixture, ingredients)
    # logging.info("TLC analysis completed.")

if __name__ == '__main__':
    logging.basicConfig(format='%(name)s -> %(funcName)s: %(message)s', level=logging.INFO)
    main()