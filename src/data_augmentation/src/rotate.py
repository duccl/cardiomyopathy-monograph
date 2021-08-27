from typing import Tuple

import cv2
import numpy as np

def by_angle(
    image: np.ndarray,
    angle: int,
    image_center: Tuple[int, int] = None,
    transformation_scale: float = 1.0
  ) -> np.ndarray:
    print('Rotating image')
    (height, width) = image.shape[:2]

    if image_center is None:
      image_center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, transformation_scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image
