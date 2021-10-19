#%%
import numpy as np
from matplotlib import pyplot as plt

from glcmbin5 import CyGLCM
BINS = 8

# np.random.seed(1)

b = CyGLCM(np.random.randint(0, 255, [500, 500, 5]).astype(np.float32), 3, BINS)
# b = CyGLCM(np.ones([100, 100, 1]).astype(np.float32), 3, BINS)
# b = CyGLCM(np.linspace(255, 1, 10000).reshape([100, 100, 1]).astype(np.float32), 3, BINS)
a = b.create_glcm()
print(f"CONTRAST    ", np.nanmin(a[...,0]),np.nanmax(a[...,0]))
print(f"CORRELATION ", np.nanmin(a[...,1]),np.nanmax(a[...,1]))
print(f"ASM         ", np.nanmin(a[...,2]),np.nanmax(a[...,2]))
print(f"MEAN        ", np.nanmin(a[...,3]),np.nanmax(a[...,3]))
print(f"VAR         ", np.nanmin(a[...,4]),np.nanmax(a[...,4]))
#%%

# CONTRAST     0.11578509 0.25999582 		(0, 49)
# CORRELATION  -0.75792825 0.7214325 		(-1, 1)
# ASM          0.0006417394 0.002728455 		(0, 1)
# MEAN         0.33418363 0.52842563 		(0, 7)
# VAR          0.060763378 0.11047482 		(0, 49)



#%%
x = np.random.randint(0, 128, [50,50])
y = np.random.randint(0, 128, [50,50])

r = np.sum((x - np.mean(x))*(y - np.mean(y))) / \
    np.sqrt(np.sum((x - np.mean(x))**2) * np.sum((y - np.mean(y))**2))

