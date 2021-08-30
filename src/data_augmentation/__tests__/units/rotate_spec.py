import cv2
import numpy as np

from ...src.rotate import by_angle
from ..mocks.image_mock import image

rotated_image = by_angle(image, 45)
(height, width) = image.shape[:2]
image_center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(image_center, 45, 1)
rotated_image_mock = cv2.warpAffine(image, rotation_matrix, (width, height))
assert np.allclose(rotated_image_mock, rotated_image), "Should rotate image correctly when image_center and scale are not provided"

rotated_image = by_angle(image, 45, None, 2)
(height, width) = image.shape[:2]
image_center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(image_center, 45, 2)
rotated_image_mock = cv2.warpAffine(image, rotation_matrix, (width, height))
assert np.allclose(rotated_image_mock, rotated_image), "Should rotate image correctly when image_center is not provided"

rotated_image = by_angle(image, 45, (1,1), 2)
(height, width) = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((1,1), 45, 2)
rotated_image_mock = cv2.warpAffine(image, rotation_matrix, (width, height))
assert np.allclose(rotated_image_mock, rotated_image), "Should rotate image correctly when all parameters are provided"
