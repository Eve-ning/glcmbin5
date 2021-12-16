
"""
This is the GLCM script I used to test how fast it runs!

Note that I ran using DIVISION = 2, that means, half the image.
This is set to 10 so that it doesn't take too long, feel free to adjust though!

- Evening
"""

#%%
from time import time

import PIL.Image
import matplotlib.pyplot as plt

from glcmbin5 import CyGLCM
import numpy as np

DIVISION = 2
FILE_NAME = "sample.jpg"

GLCM_RADIUS = 1
GLCM_BINS = 16
GLCM_STEPSIZE = 1

FIGURE_SIZE = (12,25)

s = time()
img = PIL.Image.open(FILE_NAME)
img_ar = np.asarray(img)[::DIVISION,::DIVISION]
glcm = CyGLCM(img_ar.astype(np.float32),
              radius=GLCM_RADIUS,
              bins=GLCM_BINS,
              step_size=GLCM_STEPSIZE).create_glcm()
e = time()
#%%
fig, ax = plt.subplots(5,3, sharex=True, sharey=True, figsize=FIGURE_SIZE)
ax[0][0].imshow(glcm[...,0,0], cmap='rainbow')
ax[1][0].imshow(glcm[...,0,1], cmap='rainbow')
ax[2][0].imshow(glcm[...,0,2], cmap='rainbow')
ax[3][0].imshow(glcm[...,0,3], cmap='rainbow')
ax[4][0].imshow(glcm[...,0,4], cmap='rainbow')
ax[0][1].imshow(glcm[...,1,0], cmap='rainbow')
ax[1][1].imshow(glcm[...,1,1], cmap='rainbow')
ax[2][1].imshow(glcm[...,1,2], cmap='rainbow')
ax[3][1].imshow(glcm[...,1,3], cmap='rainbow')
ax[4][1].imshow(glcm[...,1,4], cmap='rainbow')
ax[0][2].imshow(glcm[...,2,0], cmap='rainbow')
ax[1][2].imshow(glcm[...,2,1], cmap='rainbow')
ax[2][2].imshow(glcm[...,2,2], cmap='rainbow')
ax[3][2].imshow(glcm[...,2,3], cmap='rainbow')
ax[4][2].imshow(glcm[...,2,4], cmap='rainbow')
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
#%%
