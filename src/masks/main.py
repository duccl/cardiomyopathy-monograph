import numpy as np

import canny


class Masks:
  def __init__(self, image: np.ndarray = None):
    self.image = image

  def withImage(self, image: np.ndarray):
    self.image = image
    return self

  def useCanny(self):
    self.image = canny.apply(self.image)
    return self

  def apply(self) -> np.ndarray:
    return self.image
