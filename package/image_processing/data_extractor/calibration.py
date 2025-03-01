import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

def plot_all(intensity: np.ndarray, minima: np.ndarray, peak_area: np.ndarray, show_plot: bool = True):
    """
    Plot intensity, minima points, and visual rectangles illustrating calculated peak areas.
    """
    plt.plot(intensity, label='Average Intensity')
    plt.scatter(minima, intensity[minima], color='red', label='Minima')
    for index_minima in range(0, len(minima)-1, 2):
        width = minima[index_minima + 1] - minima[index_minima]
        height = peak_area[index_minima // 2] / width if width != 0 else 0
        plt.gca().add_patch(plt.Rectangle((minima[index_minima], 0), width, height, color='green', alpha=0.5))
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.title('Average Intensity')
    plt.legend()
    if show_plot:
        plt.show()

def _calculate_average_intensity(image: np.ndarray) -> np.ndarray:
    """
    Compute the average (inverted) intensity per column of a TLC image.
    The inversion is 255 - actual intensity to highlight dark pixels as high intensity.
    """
    sum_intensity = np.sum(image, axis=0)
    count_color_pixel = np.sum(np.where(image > 0, 1, 0), axis=0)
    safe_count_color_pixel = np.where(count_color_pixel == 0, 1, count_color_pixel)
    average_intensity = (255 - (sum_intensity / safe_count_color_pixel)).astype(int)
    average_intensity[count_color_pixel == 0] = 0
    return average_intensity

def _calculate_minima(intensity: np.ndarray) -> np.ndarray:
    """
    Identify indices where intensity transitions from zero to non-zero and vice versa.
    Returns sorted minima indices.
    """
    threshold_intensity = np.where(intensity > 0, 1, 0)
    zero_to_non_zero = np.where((threshold_intensity[:-1] == 0) & (threshold_intensity[1:] != 0))[0]
    non_zero_to_zero = np.where((threshold_intensity[:-1] != 0) & (threshold_intensity[1:] == 0))[0] + 1
    minima_index = np.sort(np.concatenate((zero_to_non_zero, non_zero_to_zero)))
    return minima_index

def _calculate_peak_area(intensity: np.ndarray, minima: np.ndarray) -> np.ndarray:
    """
    Calculate the integrated peak area for each pair of minima indices.
    Assumes minima has an even number of elements.
    """
    peak_area = []
    for index_minima in range(0, len(minima)-1, 2):
        peak_area.append(np.trapz(intensity[minima[index_minima]: minima[index_minima + 1] + 1]))
    return np.array(peak_area)

def _perform_linear_fit(concentration: list[float], peak_area: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Perform a linear polynomial fit between the provided concentrations and peak areas.
    Returns the polynomial coefficients and the R^2 value.
    """
    new_concentration = concentration[-len(peak_area):]
    if len(new_concentration) != len(peak_area):
        raise ValueError("The number of concentration values and peak areas are not equal.\n"
                         f"Concentration: {len(new_concentration)}\n"
                         f"Peak Area: {len(peak_area)}")
    # Use float precision for the fit coefficients
    coefs = poly.polyfit(new_concentration, peak_area, 1)  # coefs in ascending order
    # R2
    y_hat = poly.polyval(new_concentration, coefs)
    y_bar = np.mean(peak_area)
    ssr = np.sum((peak_area - y_hat) ** 2)
    sst = np.sum((peak_area - y_bar) ** 2)
    r2 = 1 - ssr/sst

    # Return reversed to match [slope, intercept] if needed
    return coefs[::-1], round(float(r2), 3)

def calculate_best_fit_line_for_image(image: np.ndarray, concentration: list[float]) -> tuple[np.ndarray, float]:
    """
    Calculate the best fit line for a given TLC image and concentration values.
    Raises a ValueError if the image is empty or if there's a mismatch between concentrations and obtained peak areas.
    Returns the best fit line coefficients and the R^2 value.
    """
    # Check if the image is empty or invalid shape
    if image.size == 0 or len(image.shape) < 2:
        raise ValueError("Invalid or empty image array.")
    intensity = _calculate_average_intensity(image)
    minima = _calculate_minima(intensity)
    # Handle odd-length minima by ignoring the last leftover index if it doesn't form a pair
    if len(minima) % 2 != 0:
        minima = minima[:-1]
    peak_area = _calculate_peak_area(intensity, minima)
    best_fit_line, r2 = _perform_linear_fit(concentration, peak_area)
    return best_fit_line, r2