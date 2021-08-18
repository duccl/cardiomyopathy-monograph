# from typing import List
# import numpy as np
# from main import DataAugmentation
# import cv2
# import time

# start_time = time.time()

# images: List[np.ndarray] = []

# _image = cv2.imread('frame01.png')
# images.append(_image)

# augmentedImages:List[np.ndarray] = []

# for image in images:
# 	augmentedImages.append(image)
# 	augmentedImages.append(DataAugmentation(image).move().apply())
# 	augmentedImages.append(DataAugmentation(image).rotate().apply())

# # for index, image in enumerate(augmentedImages):
# # 	cv2.imshow(str(index), image)

# # cv2.waitKey(0)

# print("--- %s seconds ---" % (time.time() - start_time))
