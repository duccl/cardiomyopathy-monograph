# from typing import List
# import numpy as np
# from main import DataAugmentation
# import cv2

# images:List[np.ndarray] = []

# _image = cv2.imread('image.jpg')
# images.append(_image)

# augmentedImages:List[np.ndarray] = []

# for image in images:
# 	augmentedImages.append(image)
# 	augmentedImages.append(DataAugmentation(image).rotate().move().apply())
# 	augmentedImages.append(DataAugmentation(image).move().rotate().apply())
# 	augmentedImages.append(DataAugmentation(image).move().apply())
# 	augmentedImages.append(DataAugmentation(image).rotate().apply())

# for index, image in enumerate(augmentedImages):
# 	cv2.imshow(str(index), image)

# cv2.waitKey(0)
