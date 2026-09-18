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


def _find_digit_box(binary_image: np.ndarray):
    """
    Find the component most likely to be the digit.

    A temporary dilation joins small broken parts of the handwritten digit.
    The original binary image is not permanently dilated here.
    """

    height, width = binary_image.shape
    minimum_dimension = min(height, width)

    grouping_size = _make_odd(minimum_dimension * 0.025)

    grouping_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (grouping_size, grouping_size)
    )

    grouped_image = cv2.dilate(
        binary_image,
        grouping_kernel,
        iterations=1
    )

    number_of_labels, _, stats, centroids = (
        cv2.connectedComponentsWithStats(
            grouped_image,
            connectivity=8
        )
    )

    image_center_x = (width - 1) / 2.0
    image_center_y = (height - 1) / 2.0

    best_box = None
    best_score = -1.0

    for label_number in range(1, number_of_labels):
        x = stats[label_number, cv2.CC_STAT_LEFT]
        y = stats[label_number, cv2.CC_STAT_TOP]
        box_width = stats[label_number, cv2.CC_STAT_WIDTH]
        box_height = stats[label_number, cv2.CC_STAT_HEIGHT]
        area = stats[label_number, cv2.CC_STAT_AREA]

        if area < 10:
            continue

        component_x, component_y = centroids[label_number]

        normalized_x = (
            (component_x - image_center_x) /
            max(width / 2.0, 1.0)
        )

        normalized_y = (
            (component_y - image_center_y) /
            max(height / 2.0, 1.0)
        )

        distance_squared = (
            normalized_x ** 2 +
            normalized_y ** 2
        )

        center_weight = np.exp(-2.5 * distance_squared)

        touches_border = (
            x <= 1 or
            y <= 1 or
            x + box_width >= width - 1 or
            y + box_height >= height - 1
        )

        border_weight = 0.05 if touches_border else 1.0

        score = (
            area *
            (0.35 + 0.65 * center_weight) *
            border_weight
        )

        if score > best_score:
            best_score = score
            best_box = (x, y, box_width, box_height)

    if best_box is None:
        raise ValueError(
            "No digit was found. Write a darker and larger digit."
        )

    return best_box


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
    otsu_threshold, _ = cv2.threshold(
        flattened,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    threshold_value = int(
        np.clip(otsu_threshold, 50, 215)
    )

    _, binary = cv2.threshold(
        flattened,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Join small breaks in pen strokes.
    morphology_size = _make_odd(minimum_dimension * 0.006)
    morphology_size = min(morphology_size, 9)

    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morphology_size, morphology_size)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        morphology_kernel,
        iterations=1
    )

    # Remove anything touching the extreme edge, such as shadows
    # from the edge of the paper.
    border_size = max(1, int(minimum_dimension * 0.01))

    binary[:border_size, :] = 0
    binary[-border_size:, :] = 0
    binary[:, :border_size] = 0
    binary[:, -border_size:] = 0

    minimum_component_area = max(
        8,
        int(height * width * 0.00005)
    )

    binary = _remove_small_components(
        binary,
        minimum_component_area
    )

    if cv2.countNonZero(binary) == 0:
        raise ValueError(
            "No digit was detected. Use a black pen and sufficient light."
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
            os.path.join(debug_directory, "03_binary.png"),
            binary
        )
        cv2.imwrite(
            os.path.join(debug_directory, "04_mnist.png"),
            mnist_image
        )

    # Normalize to the same 0-1 range normally used for MNIST.
    X = mnist_image.astype(np.float32) / 255.0

    return X.reshape(1, 784)