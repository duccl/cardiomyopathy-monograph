# cardiomyopathy-monograph
Monograph about Cardiomyopathy for the TCC2 course at Centro Universitário FEI

---
# data-augmentation

## Table of contents

* [`Environment`](#environment)
* [`Dependencies`](#dependencies)
* [`Running`](#running)
* [`How it works`](#how-it-works)
* [`Selected parameters`](#selected-parameters)

## Environment

![](https://img.shields.io/badge/Python-^3.9-informational?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/PIP-^21.2-informational?style=for-the-badge&logo=pypi&logoColor=white)

## Dependencies

Before install the dependencies, be sure that you have all the [`necessary environment`](#environment) configured on your local machine.

The project dependencies are:
| Name          |    | Version  |
| ------------- | -- | -------- |
| numpy         | ~> | 1.21.2   |
| opencv-python | ~> | 4.5.3.56 |

You should install the project dependencies using the following command:

```shell
$ pipenv install
```

## Running

**Important**: [`All dependencies should be installed and reached a ready state before running this feature.`](#dependencies)

Create a new file `.py` and add the following code block
```py
# ./__init.py

# import some dependencies
from main import DataAugmentation
import cv2

# read an example image
_image = cv2.imread('./image.png')

images = []
images.append(_image)
augmentedImages = []

# apply data augmentation transformations to loaded image
for image in images:
  augmentedImages.append(image)
  augmentedImages.append(DataAugmentation(image).move().apply())
  augmentedImages.append(DataAugmentation(image).rotate().apply())
  augmentedImages.append(DataAugmentation(image).rotate().move().apply())
  augmentedImages.append(DataAugmentation(image).move().rotate().apply())

# show generated images
for index, image in enumerate(augmentedImages):
  cv2.imshow(str(index), image)
cv2.waitKey(0)
```

Run the command to execute the above code block
```shell
$ python ./__init__.py
```
Then you will see the generated images in some windows

## How it works

The data augmentation feature are used to increase the volume of data that will be used to train the neural network, using the limited images base available.
We are going to apply some morphological transformations, such as rotating and moving the images

## Selected parameters

### Rotating the image
To apply a rotation around the image, we selected an angle of 45º

### Moving the image horizontally
To apply a horizontal moving to the image, we selected a value of -100 units

### Moving the image vertically
To apply a vartical moving to the image, we selected a value of 30 units
