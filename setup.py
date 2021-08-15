import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glcmbin5",
    version="0.1.0a",
    author="evening",
    author_email="dev_evening@hotmail.com",
    description="Binned Cython 5 Feature GLCM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eve-ning/glcm",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'tqdm',
        'cython',
    ]
)