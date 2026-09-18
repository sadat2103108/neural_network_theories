import os
from typing import Optional

import cv2
import numpy as np


def _make_odd(value: int) -> int:
    """Return an odd integer greater than or equal to 3."""
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _remove_small_components(
    binary_image: np.ndarray,
    minimum_area: int
) -> np.ndarray:
    """Remove small isolated noise components."""

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image,
        connectivity=8
    )

    cleaned = np.zeros_like(binary_image)

    for label_number in range(1, number_of_labels):
        area = stats[label_number, cv2.CC_STAT_AREA]

        if area >= minimum_area:
            cleaned[labels == label_number] = 255

    return cleaned



def _repair_broken_strokes(
    binary_image: np.ndarray
) -> np.ndarray:
    """
    Repair small horizontal, vertical and diagonal gaps
    without using aggressive opening.
    """

    height, width = binary_image.shape
    minimum_dimension = min(height, width)

    # General closing for small gaps in any direction.
    general_size = _make_odd(
        np.clip(
            minimum_dimension * 0.012,
            3,
            13
        )
    )

    general_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (general_size, general_size)
    )

    repaired = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        general_kernel,
        iterations=2
    )

    # Repair slightly larger horizontal and vertical breaks.
    directional_size = _make_odd(
        np.clip(
            minimum_dimension * 0.025,
            5,
            21
        )
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (directional_size, 3)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, directional_size)
    )

    horizontal_repair = cv2.morphologyEx(
        repaired,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
        iterations=1
    )

    vertical_repair = cv2.morphologyEx(
        repaired,
        cv2.MORPH_CLOSE,
        vertical_kernel,
        iterations=1
    )

    repaired = cv2.bitwise_or(
        repaired,
        horizontal_repair
    )

    repaired = cv2.bitwise_or(
        repaired,
        vertical_repair
    )

    # If the digit is extremely thin, make it slightly thicker.
    foreground_ratio = (
        cv2.countNonZero(repaired) /
        float(height * width)
    )

    if foreground_ratio < 0.035:
        thickening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        repaired = cv2.dilate(
            repaired,
            thickening_kernel,
            iterations=1
        )

    return repaired




def _find_digit_box(binary_image: np.ndarray):
    """
    Group nearby broken digit parts and find one bounding box
    containing the complete digit.
    """

    height, width = binary_image.shape
    minimum_dimension = min(height, width)

    grouping_size = _make_odd(
        np.clip(
            minimum_dimension * 0.05,
            7,
            31
        )
    )

    grouping_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (grouping_size, grouping_size)
    )

    # This image is only used for grouping components.
    grouped_image = cv2.dilate(
        binary_image,
        grouping_kernel,
        iterations=2
    )

    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            grouped_image,
            connectivity=8
        )
    )

    image_center_x = (width - 1) / 2.0
    image_center_y = (height - 1) / 2.0

    best_label = None
    best_score = -1.0

    for label_number in range(1, number_of_labels):
        x = stats[label_number, cv2.CC_STAT_LEFT]
        y = stats[label_number, cv2.CC_STAT_TOP]

        box_width = stats[
            label_number,
            cv2.CC_STAT_WIDTH
        ]

        box_height = stats[
            label_number,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label_number,
            cv2.CC_STAT_AREA
        ]

        if area < 10:
            continue

        component_x, component_y = (
            centroids[label_number]
        )

        normalized_x = (
            component_x - image_center_x
        ) / max(width / 2.0, 1.0)

        normalized_y = (
            component_y - image_center_y
        ) / max(height / 2.0, 1.0)

        distance_squared = (
            normalized_x ** 2 +
            normalized_y ** 2
        )

        center_weight = np.exp(
            -2.5 * distance_squared
        )

        touches_border = (
            x <= 1 or
            y <= 1 or
            x + box_width >= width - 1 or
            y + box_height >= height - 1
        )

        border_weight = (
            0.05 if touches_border else 1.0
        )

        score = (
            area *
            (0.30 + 0.70 * center_weight) *
            border_weight
        )

        if score > best_score:
            best_score = score
            best_label = label_number

    if best_label is None:
        raise ValueError(
            "No complete digit was found."
        )

    # Select original pixels belonging to the grouped component.
    group_mask = np.zeros_like(binary_image)

    group_mask[labels == best_label] = 255

    selected_digit = cv2.bitwise_and(
        binary_image,
        group_mask
    )

    digit_points = cv2.findNonZero(
        selected_digit
    )

    if digit_points is None:
        raise ValueError(
            "The selected digit contains no pixels."
        )

    x, y, box_width, box_height = (
        cv2.boundingRect(digit_points)
    )

    # Add a little padding around the complete digit.
    padding = max(
        2,
        int(max(box_width, box_height) * 0.06)
    )

    x = max(0, x - padding)
    y = max(0, y - padding)

    right = min(
        width,
        x + box_width + 2 * padding
    )

    bottom = min(
        height,
        y + box_height + 2 * padding
    )

    return x, y, right - x, bottom - y




def _place_on_mnist_canvas(
    digit_image: np.ndarray
) -> np.ndarray:
    """
    Resize the digit into a 20x20 area, place it on a 28x28 canvas,
    and center it using its center of mass.
    """

    height, width = digit_image.shape

    if height == 0 or width == 0:
        raise ValueError("The detected digit region is empty.")

    scale = 20.0 / max(height, width)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized_digit = cv2.resize(
        digit_image,
        (new_width, new_height),
        interpolation=interpolation
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    start_x = (28 - new_width) // 2
    start_y = (28 - new_height) // 2

    canvas[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized_digit

    moments = cv2.moments(canvas)

    if moments["m00"] == 0:
        raise ValueError("The processed digit contains no foreground pixels.")

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]

    shift_x = 13.5 - center_x
    shift_y = 13.5 - center_y

    transformation_matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    centered_digit = cv2.warpAffine(
        canvas,
        transformation_matrix,
        (28, 28),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return centered_digit


def image_to_mnist(
    image_path: str,
    debug_directory: Optional[str] = None
) -> np.ndarray:
    """
    Convert a camera image containing black handwriting on white paper
    into an MNIST-style input.

    Output:
        Shape: (1, 784)
        Type: float32
        Range: 0.0 to 1.0
        Background: black
        Digit: white
    """

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Handle transparent PNG images.
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        color = image[:, :, :3].astype(np.float32)

        image = (
            color * alpha[:, :, None] +
            255.0 * (1.0 - alpha[:, :, None])
        ).astype(np.uint8)

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Large camera frames do not need to be processed at full resolution.
    maximum_dimension = max(gray.shape)

    if maximum_dimension > 1200:
        resize_scale = 1200.0 / maximum_dimension

        gray = cv2.resize(
            gray,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA
        )

    height, width = gray.shape
    minimum_dimension = min(height, width)

    # Remove small camera noise.
    gray = cv2.medianBlur(gray, 3)

    # Estimate the paper illumination. This helps remove shadows and
    # uneven lighting across the page.
    illumination_sigma = max(5.0, minimum_dimension / 18.0)

    paper_background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=illumination_sigma,
        sigmaY=illumination_sigma
    )

    # Flatten the illumination. White paper becomes approximately 255,
    # while dark pen strokes remain dark.
    flattened = cv2.divide(
        gray,
        paper_background,
        scale=255
    )

    flattened = cv2.GaussianBlur(
        flattened,
        (3, 3),
        0
    )

    # Otsu chooses a threshold automatically. Limiting the threshold
    # prevents light paper texture from becoming foreground.
    # Global Otsu threshold.
    otsu_threshold, _ = cv2.threshold(
        flattened,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    threshold_value = int(
        np.clip(
            otsu_threshold,
            50,
            230
        )
    )

    _, global_binary = cv2.threshold(
        flattened,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Adaptive threshold preserves faint or shadowed stroke areas.
    adaptive_block_size = _make_odd(
        np.clip(
            minimum_dimension * 0.15,
            31,
            151
        )
    )

    adaptive_binary = cv2.adaptiveThreshold(
        flattened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        8
    )

    # Prevent very light paper texture from being selected.
    dark_pixel_mask = np.where(
        flattened < 238,
        255,
        0
    ).astype(np.uint8)

    adaptive_binary = cv2.bitwise_and(
        adaptive_binary,
        dark_pixel_mask
    )

    # Combine strong and faint parts of the pen stroke.
    binary_before_repair = cv2.bitwise_or(
        global_binary,
        adaptive_binary
    )

    # Remove border shadows before performing closing.
    border_size = max(
        1,
        int(minimum_dimension * 0.01)
    )

    binary_before_repair[:border_size, :] = 0
    binary_before_repair[-border_size:, :] = 0
    binary_before_repair[:, :border_size] = 0
    binary_before_repair[:, -border_size:] = 0

    # Repair broken strokes.
    binary = _repair_broken_strokes(
        binary_before_repair
    )

    # Only remove extremely small noise.
    minimum_component_area = max(
        4,
        int(height * width * 0.000015)
    )

    binary = _remove_small_components(
        binary,
        minimum_component_area
    )

    if cv2.countNonZero(binary) == 0:
        raise ValueError(
            "No digit was detected. Use a darker pen."
        )
            
        
        

    x, y, box_width, box_height = _find_digit_box(binary)

    digit = binary[
        y:y + box_height,
        x:x + box_width
    ]

    mnist_image = _place_on_mnist_canvas(digit)

    # Optional debug images.
    if debug_directory is not None:
        os.makedirs(debug_directory, exist_ok=True)

        cv2.imwrite(
            os.path.join(debug_directory, "01_gray.png"),
            gray
        )
        cv2.imwrite(
            os.path.join(debug_directory, "02_flattened.png"),
            flattened
        )
        
        cv2.imwrite(
            os.path.join(
                debug_directory,
                "03_before_repair.png"
            ),
            binary_before_repair
        )

        cv2.imwrite(
            os.path.join(
                debug_directory,
                "04_after_repair.png"
            ),
            binary
        )

        cv2.imwrite(
            os.path.join(
                debug_directory,
                "05_final_mnist.png"
            ),
            mnist_image
        )

    # Normalize to the same 0-1 range normally used for MNIST.
    X = mnist_image.astype(np.float32) / 255.0

    return X.reshape(1, 784)