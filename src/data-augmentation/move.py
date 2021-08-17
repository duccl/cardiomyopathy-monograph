import cv2
from numpy import array, ndarray


def moveVertically(image: ndarray, value: int) -> ndarray:
	print('Moving image vertically')
	(height, width) = image.shape[:2]

	translationMatrix = array([[1, 0, 1], [0, 1, height * value]])

	# Apply translation based on translation matrix
	translatedImage = cv2.warpAffine(image, translationMatrix, (width, height))

	return translatedImage


def moveHorizontally(image: ndarray, value: int) -> ndarray:
	print('Moving image horizontally')
	(height, width) = image.shape[:2]

	translationMatrix = array([[1, 0, width * value], [0, 1, 1]])

	# Apply translation based on translation matrix
	translatedImage = cv2.warpAffine(image, translationMatrix, (width, height))

	return translatedImage
