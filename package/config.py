import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass, field

@dataclass
class GaussianBlurConfig:
    """
    Configuration for Gaussian Blur.

    Attributes:
        kernel_size (Tuple[int, int]): Size of the Gaussian kernel.
        sigma_x (int): Gaussian kernel standard deviation in X direction.
    """
    kernel_size: Tuple[int, int] = (7, 7)
    sigma_x: int = 0

    def __post_init__(self):
        assert self.kernel_size[0] % 2 == 1, "Kernel size must be odd"

@dataclass
class ThresholdingConfig:
    """
    Configuration for Thresholding.

    Attributes:
        threshold_value (int): Threshold value.
        max_value (int): Maximum value to use with the THRESH_BINARY thresholding type.
        type (int): Thresholding type.
    """
    threshold_value: int = 0
    max_value: int = 255
    type: int = cv2.THRESH_BINARY + cv2.THRESH_OTSU

@dataclass
class ContoursConfig:
    """
    Configuration for Contour Detection.

    Attributes:
        contours_retrieval_mode (int): Contour retrieval mode.
        contours_approximation_method (int): Contour approximation method.
    """
    contours_retrieval_mode: int = cv2.RETR_EXTERNAL
    contours_approximation_method: int = cv2.CHAIN_APPROX_SIMPLE

@dataclass
class CropConfig:
    """
    Configuration for Cropping.

    Attributes:
        gaussian_blur (GaussianBlurConfig): Configuration for Gaussian Blur.
        thresholding (ThresholdingConfig): Configuration for Thresholding.
        contours (ContoursConfig): Configuration for Contour Detection.
    """
    gaussian_blur: GaussianBlurConfig = field(default_factory=GaussianBlurConfig)
    thresholding: ThresholdingConfig = field(default_factory=ThresholdingConfig)
    contours: ContoursConfig = field(default_factory=ContoursConfig)

@dataclass
class CannyConfig:
    """
    Configuration for Canny Edge Detection.

    Attributes:
        lower_threshold (int): Lower threshold for the hysteresis procedure.
        upper_threshold (int): Upper threshold for the hysteresis procedure.
        use_l2_gradient (bool): Whether to use L2 gradient.
        auto_threshold_ratio (Tuple[float, float]): Ratio for automatic thresholding.
    """
    lower_threshold: int = 50
    upper_threshold: int = 150
    use_l2_gradient: bool = True
    auto_threshold_ratio: Tuple[float, float] = (0.05, 0.15)

    def __post_init__(self):
        assert 0 <= self.lower_threshold <= 255, "Invalid lower threshold (0-255)"
        assert 0 <= self.upper_threshold <= 255, "Invalid upper threshold (0-255)"
        assert self.lower_threshold < self.upper_threshold, "Lower threshold must be < upper"
        assert all(0 <= r <= 1 for r in self.auto_threshold_ratio), "Auto ratios must be 0-1"

@dataclass
class HoughLinesConfig:
    """
    Configuration for Hough Line Detection.

    Attributes:
        rho (int): Distance resolution of the accumulator in pixels.
        theta (float): Angle resolution of the accumulator in radians.
        adaptive_threshold (float): Adaptive threshold for line detection.
        min_theta (float): Minimum angle to check for lines.
        max_theta (float): Maximum angle to check for lines.
    """
    rho: int = 1
    theta: float = np.pi/180
    adaptive_threshold: float = 0.5
    min_theta: float = (np.pi / 2) - np.deg2rad(5)
    max_theta: float = (np.pi / 2) + np.deg2rad(5)

    def __post_init__(self):
        assert self.rho > 0, "Rho must be positive"
        assert 0 < self.theta <= np.pi, "Theta must be in the range (0, pi]"
        assert 0 < self.adaptive_threshold < 1, "Threshold range must be within (0, 1]"
        assert 0 <= self.min_theta < self.max_theta <= np.pi, "Theta range must be within (0, pi]"

@dataclass
class FilterConfig:
    """
    Configuration for Filtering.

    Attributes:
        min_border_distance (int): Minimum distance from the border.
    """
    min_border_distance: int = 20

@dataclass
class DrawConfig:
    """
    Configuration for Drawing.

    Attributes:
        line_color (Tuple[int, int, int]): Color of the line.
        line_thickness (int): Thickness of the line.
    """
    line_color: Tuple[int, int, int] = (0, 0, 255)
    line_thickness: int = 1

@dataclass
class LineDetectionConfig:
    """
    Configuration for Line Detection.

    Attributes:
        canny (CannyConfig): Configuration for Canny Edge Detection.
        hough_lines (HoughLinesConfig): Configuration for Hough Line Detection.
        filter (FilterConfig): Configuration for Filtering.
        draw (DrawConfig): Configuration for Drawing.
    """
    canny: CannyConfig = field(default_factory=CannyConfig)
    hough_lines: HoughLinesConfig = field(default_factory=HoughLinesConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    draw: DrawConfig = field(default_factory=DrawConfig)

@dataclass
class SegmentCalibrationConfig:
    """
    Configuration for Segment Calibration.

    Attributes:
        min_contour_area (int): Minimum contour area.
    """
    min_contour_area: int = 500

@dataclass
class ThresholdGaussianBlurConfig:
    """
    Configuration for Gaussian Blur in Thresholding.

    Attributes:
        kernel_size (Tuple[int, int]): Size of the Gaussian kernel.
        sigma_x (int): Gaussian kernel standard deviation in X direction.
    """
    kernel_size: Tuple[int, int] = (11, 11)
    sigma_x: int = 0

    def __post_init__(self):
        assert self.kernel_size[0] % 2 == 1, "Kernel size must be odd"

@dataclass
class CLAHEConfig:
    """
    Configuration for CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Attributes:
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.
    """
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (11, 11)

@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for Adaptive Thresholding.

    Attributes:
        max_value (int): Maximum value to use with the THRESH_BINARY thresholding type.
        adaptive_method (int): Adaptive thresholding algorithm to use.
        threshold_type (int): Thresholding type.
        block_size (int): Size of a pixel neighborhood used to calculate a threshold value.
        constant (int): Constant subtracted from the mean or weighted mean.
        block_count_calibration (int): Block count for calibration.
        block_count_mixture (int): Block count for mixture.
    """
    max_value: int = 255
    adaptive_method: int = cv2.ADAPTIVE_THRESH_MEAN_C
    threshold_type: int = cv2.THRESH_BINARY_INV
    block_size: int = 151
    constant: int = 10
    block_count_calibration: int = 11
    block_count_mixture: int = 3

@dataclass
class MorphConfig:
    """
    Configuration for Morphological Transformations.

    Attributes:
        kernel_size (Tuple[int, int]): Size of the morphological kernel.
    """
    kernel_size: Tuple[int, int] = (25, 25)

@dataclass
class DrawContoursConfig:
    """
    Configuration for Drawing Contours.

    Attributes:
        contour_index (int): Index of the contour to draw.
        color (Tuple[int, int, int]): Color of the contour.
        thickness (int): Thickness of the contour line.
        line_type (int): Type of the line.
    """
    contour_index: int = -1 # All
    color: Tuple[int, int, int] = (0, 255, 255)
    thickness: int = 4
    line_type: int = cv2.LINE_AA

@dataclass
class DrawBoundingBoxConfig:
    """
    Configuration for Drawing Bounding Boxes.

    Attributes:
        rectangle_color (Tuple[int, int, int]): Color of the rectangle.
        rectangle_thickness (int): Thickness of the rectangle.
        text_font (int): Font of the text.
        text_scale (int): Scale of the text.
        text_color (Tuple[int, int, int]): Color of the text.
        text_thickness (int): Thickness of the text.
    """
    rectangle_color: Tuple[int, int, int] = (255, 255, 255)
    rectangle_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_thickness: int = 2

@dataclass
class CropVerticallyConfig:
    """
    Configuration for Vertical Cropping.

    Attributes:
        min_area (int): Minimum area for cropping.
    """
    min_area: int = 800

@dataclass
class CropHorizontallyConfig:
    """
    Configuration for Horizontal Cropping.

    Attributes:
        min_area (int): Minimum area for cropping.
    """
    min_area: int = 800

@dataclass
class ThresholdConfig:
    """
    Configuration for Thresholding.

    Attributes:
        segment_calibration (SegmentCalibrationConfig): Configuration for Segment Calibration.
        gaussian_blur (ThresholdGaussianBlurConfig): Configuration for Gaussian Blur.
        clahe (CLAHEConfig): Configuration for CLAHE.
        adaptive_threshold (AdaptiveThresholdConfig): Configuration for Adaptive Thresholding.
        morph (MorphConfig): Configuration for Morphological Transformations.
        draw_contours (DrawContoursConfig): Configuration for Drawing Contours.
        draw_bounding_box (DrawBoundingBoxConfig): Configuration for Drawing Bounding Boxes.
        crop_vertically (CropVerticallyConfig): Configuration for Vertical Cropping.
        crop_horizontally (CropHorizontallyConfig): Configuration for Horizontal Cropping.
    """
    segment_calibration: SegmentCalibrationConfig = field(default_factory=SegmentCalibrationConfig)
    gaussian_blur: ThresholdGaussianBlurConfig = field(default_factory=ThresholdGaussianBlurConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    adaptive_threshold: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    morph: MorphConfig = field(default_factory=MorphConfig)
    draw_contours: DrawContoursConfig = field(default_factory=DrawContoursConfig)
    draw_bounding_box: DrawBoundingBoxConfig = field(default_factory=DrawBoundingBoxConfig)
    crop_vertically: CropVerticallyConfig = field(default_factory=CropVerticallyConfig)
    crop_horizontally: CropHorizontallyConfig = field(default_factory=CropHorizontallyConfig)

# Instantiate the configurations
crop_config = CropConfig()
line_detection_config = LineDetectionConfig()
threshold_config = ThresholdConfig()