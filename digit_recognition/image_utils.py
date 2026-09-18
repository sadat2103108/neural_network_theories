import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_to_mnist(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("Could not find a digit in the image.")

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_index = np.argmax(areas) + 1

    largest_area = stats[
        largest_index,
        cv2.CC_STAT_AREA
    ]

    if largest_area < 20:
        raise ValueError("Could not find a sufficiently large digit.")

    x = stats[largest_index, cv2.CC_STAT_LEFT]
    y = stats[largest_index, cv2.CC_STAT_TOP]
    w = stats[largest_index, cv2.CC_STAT_WIDTH]
    h = stats[largest_index, cv2.CC_STAT_HEIGHT]

    digit = binary[y:y+h, x:x+w]

    size = max(w, h)

    padding = int(size * 0.20)

    size += 2 * padding

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    offset_x = (size - w) // 2
    offset_y = (size - h) // 2

    square[
        offset_y:offset_y+h,
        offset_x:offset_x+w
    ] = digit

    mnist_image = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    moments = cv2.moments(mnist_image)

    if moments["m00"] != 0:

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        M = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        mnist_image = cv2.warpAffine(
            mnist_image,
            M,
            (28, 28),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    mnist_image = mnist_image.astype(np.float32) / 255.0

    X = mnist_image.reshape(1, 784)

    return X


def showImg(X,title=-1):

    X = X.reshape(28, 28)

    plt.imshow(X, cmap="gray")
    plt.title(f"Label: {title}")

    plt.axis("off")

    plt.show()
