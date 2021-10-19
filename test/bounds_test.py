import numpy as np

from glcm.glcm import CyGLCM



# content of test_sample.py
def inc(x):
    return x + 1


def test_answer():
    a = CyGLCM(np.random.randint(0, 255, [100, 100, 5]), np.float32).create_glcm()