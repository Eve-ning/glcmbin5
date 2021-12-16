
"""
This is the GLCM script I used to test how fast it runs on Gray Scale!

Note that the script only accepts 3-dim arrays, which means, your gray
scale images must be shape = (xxx, yyy, 1)

If yours is (xxx, yyy), just add a dimension using ar[..., np.newaxis]

If you're slicing from an RGB image, you slice a 1 thick range [..., 0:1] just as a shortcut

- Evening
"""

#%%
from time import time

import PIL.Image
import matplotlib.pyplot as plt

from glcmbin5 import CyGLCM
import numpy as np

DIVISION = 10
FILE_NAME = "sample.jpg"

GLCM_RADIUS = 1
GLCM_BINS = 16
GLCM_STEPSIZE = 1

FIGURE_SIZE = (12,25)

s = time()
img = PIL.Image.open(FILE_NAME)
img_ar = np.asarray(img)[::DIVISION,::DIVISION, 0][..., np.newaxis]
# OR slice channels instead of index
# img_ar = np.asarray(img)[::DIVISION,::DIVISION, 0:1]

glcm = CyGLCM(img_ar.astype(np.float32),
              radius=GLCM_RADIUS,
              bins=GLCM_BINS,
              step_size=GLCM_STEPSIZE).create_glcm()
e = time()
#%%
fig, ax = plt.subplots(5, sharex=True, sharey=True, figsize=FIGURE_SIZE)
ax[0].imshow(glcm[...,0,0], cmap='rainbow')
ax[1].imshow(glcm[...,0,1], cmap='rainbow')
ax[2].imshow(glcm[...,0,2], cmap='rainbow')
ax[3].imshow(glcm[...,0,3], cmap='rainbow')
ax[4].imshow(glcm[...,0,4], cmap='rainbow')
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
#%%
