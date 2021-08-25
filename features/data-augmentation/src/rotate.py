from typing import Tuple

import cv2
import numpy as np

def byAngle(
    image: np.ndarray,
    angle: int,
    imageCenter: Tuple[int, int] = None,
    transformationScale: float = 1.0
  ) -> np.ndarray:
    print('Rotating image')
    (height, width) = image.shape[:2]

    if imageCenter is None:
      imageCenter = (width / 2, height / 2)

    rotationMatrix = cv2.getRotationMatrix2D(imageCenter, angle, transformationScale)

    # Apply rotation based on rotation matrix
    rotatedImage = cv2.warpAffine(image, rotationMatrix, (width, height))

    return rotatedImage