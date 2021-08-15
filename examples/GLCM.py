#%%
from time import time

import PIL.Image
import matplotlib.pyplot as plt

from glcm.glcm import CyGLCM
import numpy as np

s = time()
img = PIL.Image.open("sample.jpg")
img_ar = np.asarray(img)[::2, ::2]
glcm = CyGLCM(img_ar.astype(np.float32), radius=3, bins=8,
              pairs=('H','V','SE','NE')).create_glcm()
e = time()
#%%
fig, ax = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(12,12))
for i in range(5):
    for j in range(3):
        ax[i][j].imshow(glcm[...,j,i])
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
