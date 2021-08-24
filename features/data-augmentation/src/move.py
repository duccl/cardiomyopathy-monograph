import cv2
import numpy as np

def vertically(image: np.ndarray, value: int) -> np.ndarray:
  print('Moving image vertically')
  (height, width) = image.shape[:2]

  translationMatrix = np.float32([[1, 0, 1], [0, 1, value]])

  # Apply translation based on translation matrix
  translatedImage = cv2.warpAffine(image, translationMatrix, (width, height))

  return translatedImage


def horizontally(image: np.ndarray, value: int) -> np.ndarray:
  print('Moving image horizontally')
  (height, width) = image.shape[:2]

  translationMatrix = np.float32([[1, 0, value], [0, 1, 1]])

  # Apply translation based on translation matrix
  translatedImage = cv2.warpAffine(image, translationMatrix, (width, height))

  return translatedImage
