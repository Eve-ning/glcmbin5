
import time

import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
#%%
s = time.time()
for i in range(10000):
    glcm = greycomatrix(np.random.randint(0, 16, [3, 3]).astype(np.uint8), [1], [0])
    g = greycoprops(glcm, 'contrast')
    g = greycoprops(glcm, 'dissimilarity')
    g = greycoprops(glcm, 'energy')
    g = greycoprops(glcm, 'ASM')
    g = greycoprops(glcm, 'correlation')

e = time.time()

# / 10000 for each window
# 1111 * 1986 because the test image has that many windows
# 3 for 3 channels
# * 8 for 8 directions
# / (6 * 60 + 17) test timing
# ~ 505.21145648426085 for me
print(f"Optimization: {((e - s) * 8 / 10000 * 1111 * 1986 * 3) / (6 * 60 + 17):.2%}")

#%%
((e - s) * 8 / 10000 * 1111 * 1986 * 3) / 60 / 60

