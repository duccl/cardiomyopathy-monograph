import numpy as np

image = np.zeros([5,5,3])

image[:,:,0] = np.ones([5,5])*64/255.0
image[:,:,1] = np.ones([5,5])*128/255.0
image[:,:,2] = np.ones([5,5])*192/255.0
