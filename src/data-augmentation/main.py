from numpy import ndarray

import move
import rotate


class DataAugmentation:
  def __init__(self, image: ndarray = None):
    self.image = image

  def withImage(self, image: ndarray):
    self.image = image

    return self

  def rotate(self):
    self.image = rotate.byAngle(self.image, 45)

    return self

  def move(self):
    self.image = move.moveHorizontally(self.image, 0.5)
    self.image = move.moveVertically(self.image, 0.75)

    return self

  def apply(self) -> ndarray:
    return self.image
