import numpy as np

import formula


def transform_image_with_qsigmoid(image):
  result = np.zeros(image.shape)
  result = formula.qsigmoid(image = image, beta = 2, alfa = 0.51, q = .1, lamb = 0.41)
  return result
