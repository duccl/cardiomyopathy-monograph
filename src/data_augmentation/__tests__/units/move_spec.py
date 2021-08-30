import cv2
import numpy as np

from ...src.move import horizontally, vertically
from ..mocks.image_mock import image

tranlated_image = horizontally(image, -100)
(height, width) = image.shape[:2]
translation_matrix = np.float32([[1, 0, -100], [0, 1, 1]])
translated_image_mock = cv2.warpAffine(image, translation_matrix, (width, height))
assert np.allclose(translated_image_mock, tranlated_image), "Should translate image horizontally with success"

tranlated_image = vertically(image, 30)
(height, width) = image.shape[:2]
translation_matrix = np.float32([[1, 0, 30], [0, 1, 1]])
translated_image_mock = cv2.warpAffine(image, translation_matrix, (width, height))
assert np.allclose(translated_image_mock, tranlated_image), "Should translate image vertically with success"
