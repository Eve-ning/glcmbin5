from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='glcm',
    ext_modules=cythonize("glcm/**/*.pyx", build_dir="cython-build",
                          annotate=True),
    include_dirs=[numpy.get_include()]
)
