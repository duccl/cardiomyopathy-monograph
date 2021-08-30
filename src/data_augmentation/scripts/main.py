import numpy as np

import move
import rotate

class DataAugmentation:
  def __init__(self, image: np.ndarray = None):
    self.image = image

  def with_image(self, image: np.ndarray):
    self.image = image
    return self

  def rotate(self):
    self.image = rotate.by_angle(self.image, 45)
    return self

  def move(self):
    self.image = move.horizontally(self.image, -100)
    self.image = move.vertically(self.image, 30)
    return self

  def apply(self) -> np.ndarray:
    return self.image
