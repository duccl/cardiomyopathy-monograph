import cv2
import numpy as np

from ...src.rotate import byAngle
from ..mocks.image_mock import image

rotated_image = byAngle(image, 45)
(height, width) = image.shape[:2]
imageCenter = (width / 2, height / 2)
rotationMatrix = cv2.getRotationMatrix2D(imageCenter, 45, 1)
rotated_image_mock = cv2.warpAffine(image, rotationMatrix, (width, height))
assert np.allclose(rotated_image_mock, rotated_image), "Should rotate image correctly when imageCenter and scale are not provided"

rotated_image = byAngle(image, 45, None, 2)
(height, width) = image.shape[:2]
imageCenter = (width / 2, height / 2)
rotationMatrix = cv2.getRotationMatrix2D(imageCenter, 45, 2)
rotated_image_mock = cv2.warpAffine(image, rotationMatrix, (width, height))
assert np.allclose(rotated_image_mock, rotated_image), "Should rotate image correctly when imageCenter is not provided"

rotated_image = byAngle(image, 45, (1,1), 2)
(height, width) = image.shape[:2]
rotationMatrix = cv2.getRotationMatrix2D((1,1), 45, 2)
rotated_image_mock = cv2.warpAffine(image, rotationMatrix, (width, height))
assert np.allclose(rotated_image_mock, rotated_image), "Should rotate image correctly when all parameters are provided"
