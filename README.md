# Binned Cython 5 Feature GLCM

## Notice

This project has mainly migrated to https://github.com/Eve-ning/glcm-cupy which has many more new features & optimization improvements.

This repository will be left here for archival purposes. 

![Result](result.jpg)

[**Photo by Flo Maderebner from Pexels**](https://www.pexels.com/@fmaderebner)

400 Times faster, the above image takes 6+ Days to process, compared to 24 Minutes with `glcmbin5` with `bins=16`. YMMV

## Installation

I recommend to install this via forking. The `pip` version may be unstable.

Run this command with the `c_setup.py` here
```
python c_setup.py build_ext --inplace
```

## Main Features

- Implemented in `Cython`, means you get C-Lang Speed on Python
- 2/4/6/8-directional GLCM, in the case where orientation doesn't matter
- Memory Optimized, calculations are **per window**, thus a large intermediate GLCM isn't used, saving GBs of memory.

**Pitfalls**

This doesn't work with parallel processing for some reason, I couldn't get it to work unfortunately.

If you derived this work to make it parallel, please let me know! I would love to use it.

### Speed

This is  ~400 times faster than using ``skimage.feature.graycomatrix`` and ``graycoprops``
because this is Cython optimized, YMMV.

With a *2000x1000x3* image, it takes around 6+ minutes.
Compared to **39 hours** with ``skimage``

```
GLCM Progress: 100%|██████████| 12/12 [06:49<00:00, 34.09s/it]
```

### Memory Size

**Binning** the input array reduces size of GLCM, also increases performance.
Though there's no substantial evidence of it improving nor deproving results.

With this algorithm, I omit generating the whole GLCM, instead, it's integrated in
the GLCM feature calculation. Memory used is freed asap.

## Example

You can also see an example in [`/examples`](https://github.com/Eve-ning/glcm/tree/master/examples)

```python
from glcm.glcm import CyGLCM
import numpy as np
ar = ...
glcm = CyGLCM(ar.astype(np.float32),
              radius=3,
              bins=8,
              pairs=('H', 'V', 'SE', 'NE')
              ).create_glcm()
```

## Arguments

- **radius**: The radius of the GLCM window
- **bins**: The number of bins to use
- **pairs**:
  - **H**: Horizontal Pair
  - **V**: Vertical Pair
  - **SE**: South-East Diagonal Pair
  - **NE**: North-East Diagonal Pair

## I/O

**Input**:
- `ndim = 3`
- `shape=(in_dim0, in_dim1, channel)`

**Output**:
- `ndim = 4`
- `shape=(in_dim0, in_dim1, channel, features)`
- Methods:
  - Homogeneity
  - Correlation
  - Angular Second Moment
  - GLCM Mean
  - GLCM Variance

## Progress Bar

The progress bar value is the current pair calculated.

## Gotchas

### GLCM Shrink

The resulting GLCM array will be smaller than the original.

*GLCM Dimension = Dimension - (2 * radius + step_size) = Dimension - Diameter*

### Data Type `float32`

Arrays **MUST BE** in ``np.float32``, you need to cast it.
```
ar.astype(np.float32)
```

## Features

Based on [GLCM Texture: A Tutorial v. 3.0 March 2017](https://prism.ucalgary.ca/handle/1880/51900).

For an effective segmentation, we just need 5 features as selected here.

Many features are not significantly orthogonal, hence more will introduce redundancy.

Chosen methods are for simplicity and efficiency in coding. 

``0.1.5`` swapped Contrast for Homogeneity as it may be more orthogonal.

## Binning

Arrays are **Binned** before going through GLCM.

All arrays will be processed to integer values `[0,bin-1]` band-independently.

# Citation

If you have used or referenced any of the code in the repository,
please kindly cite

```
@misc{glcmbin5,
  author = {John Chang},
  title = {Binned Cython 5 Feature GLCM},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Eve-ning/glcmbin5}},
}
```

# Acknowledgements

- [Dr. Ji-Jon Sit](https://dr.ntu.edu.sg/cris/rp/rp00175) for hosting the parent project [`Eve-ning/FRModel`](https://github.com/Eve-ning/FRModel) which lead to this algorithm
- [Wang Ji Fei](https://fass.nus.edu.sg/geog/people/wang-jifei/) for discovering GLCM Binning optimization.
- [GLCM Texture: A Tutorial v. 3.0 March 2017](https://prism.ucalgary.ca/handle/1880/51900) for providing a
simple tutorial to guide this implementation.


