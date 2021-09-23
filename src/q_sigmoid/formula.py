import numpy as np

def qsigmoid(image, beta = 0.3, alfa = 0.1, lamb = 0.5, q = 2):
    tsallis = 1 / (1 - q)

    if q < 1:
        local_intensity = np.abs(image - beta) / alfa
        base = (1.0 + (lamb * (1 - q) * (local_intensity)))
        gaussian = base ** tsallis

        return 2.0 / (1.0 + gaussian)

    image[image == beta] = 1

    local_intensity = -1 * (1 / (np.abs(image - beta) / alfa ))
    base = (1.0 + (lamb * (1 - q) * (local_intensity)))
    gaussian = base ** tsallis

    return 1.0 / (1.0 + gaussian)
