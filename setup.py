import numpy as np
import setuptools
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glcmbin5",
    version="0.1.5",
    author="evening",
    author_email="dev_evening@hotmail.com",
    description="Binned Cython 5 Feature GLCM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eve-ning/glcm",
    ext_modules=cythonize("**/*.pyx", include_path=["src"]),
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["cython", "numpy"],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'tqdm',
        'cython',
    ],
    include_dirs=np.get_include(),
    include_package_data=True,
)