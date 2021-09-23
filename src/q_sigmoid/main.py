import numpy as np

import formula


def transformImageWithQSigmoid(image):
  result = np.zeros(image.shape)
  result = formula.q_sigmoid(image = image, beta = 2, alfa = 0.51, q = .1, lamb = 0.41)
  return result
