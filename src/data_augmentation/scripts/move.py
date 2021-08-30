import cv2
import numpy as np

def vertically(image: np.ndarray, value: int) -> np.ndarray:
  print('Moving image vertically')
  (height, width) = image.shape[:2]

  translation_matrix = np.float32([[1, 0, 1], [0, 1, value]])
  translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

  return translated_image


def horizontally(image: np.ndarray, value: int) -> np.ndarray:
  print('Moving image horizontally')
  (height, width) = image.shape[:2]

  translation_matrix = np.float32([[1, 0, value], [0, 1, 1]])
  translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

  return translated_image
