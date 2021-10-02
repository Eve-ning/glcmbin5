#%%
import numpy as np
from matplotlib import pyplot as plt

from glcm.glcm import CyGLCM
a = CyGLCM(np.zeros([7, 7, 1]).astype(np.float32), 1, 2, True, ('NE',))
b = a.create_glcm()

#%%
# a = CyGLCM(np.random.randint(0, 255, [100, 100, 5]).astype(np.float32), 2, 8).create_glcm()

#%%
a.glcm

#%%
fig, ax = plt.subplots(5, figsize=(2,10))

for i in range(5):
    ax[i].imshow(a[...,:3,i])
fig.show()
