"""
    GLCM in Cython
    Copyright (C) 2021  John Chang / Eve-ning

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
cimport numpy as np
cimport cython
from skimage.util import view_as_windows
from libc.math cimport sqrt
ctypedef np.uint8_t DTYPE_t8
ctypedef np.uint16_t DTYPE_t16
ctypedef np.uint32_t DTYPE_t32
ctypedef np.float32_t DTYPE_ft32
from tqdm import tqdm
from libc.math cimport sqrt


cdef enum:
    CONTRAST = 0
    CORRELATION = 1
    ASM = 2
    MEAN = 3
    VAR = 4

cdef class CyGLCM:
    cdef public DTYPE_t8 radius, bins, diameter
    cdef public np.ndarray ar
    cdef public np.ndarray features
    cdef public np.ndarray glcm
    cdef public tuple pairs
    cdef public char verbose

    def __init__(self, np.ndarray[DTYPE_ft32, ndim=3] ar,
                 DTYPE_t8 radius, DTYPE_t8 bins,
                 verbose=True,
                 pairs=('H', 'V', 'SE', 'NE')):
        self.radius = radius
        self.diameter = radius * 2 + 1
        self.bins = bins
        self.ar = ar
        self.features = np.zeros([ar.shape[0] - self.diameter,
                                  ar.shape[1] - self.diameter,
                                  ar.shape[2], 5],
                                 dtype=np.float32)
        self.glcm = np.zeros([bins, bins], dtype=np.uint8)
        self.pairs = pairs
        self.verbose = verbose

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_glcm(self):
        cdef np.ndarray[DTYPE_ft32, ndim=3] ar = self.ar
        cdef np.ndarray[DTYPE_ft32, ndim=4] features = self.features

        cdef np.ndarray ar_bin = self._binarize(ar)
        cdef DTYPE_t8 chs = ar_bin.shape[2]
        with tqdm(total=chs * len(self.pairs), disable=not self.verbose,
                  desc=f"GLCM Progress") as pbar:
            for ch in range(chs):
                pairs = self._pair(ar_bin[..., ch])
                for pair in pairs:
                    # Each pair is a tuple
                    # Tuple of 2 offset images for GLCM calculation.
                    self._populate_glcm(pair[0], pair[1], features[:,:,ch,:])
                    pbar.update()

        return self.features

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _populate_glcm(self,
                       np.ndarray[DTYPE_t8, ndim=4] pair_i,
                       np.ndarray[DTYPE_t8, ndim=4] pair_j,
                       np.ndarray[DTYPE_ft32, ndim=3] features):
        """ The ar would be WR, WC, CR, CC

        :param pair_i: WR WC CR CC
        :param pair_j: WR WC CR CC
        :return:
        """
        cdef DTYPE_t16 wrs = pair_i.shape[0]
        cdef DTYPE_t16 wcs = pair_i.shape[1]
        cdef DTYPE_t16 wr, wc;

        for wr in range(wrs):
            for wc in range(wcs):
                self._populate_glcm_single(pair_i[wr, wc],
                                           pair_j[wr, wc],
                                           features[wr, wc])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _populate_glcm_single(self,
                              np.ndarray[DTYPE_t8, ndim=2] pair_i,
                              np.ndarray[DTYPE_t8, ndim=2] pair_j,
                              np.ndarray[DTYPE_ft32, ndim=1] features):
        """

        :param pair_i: CR CC
        :param pair_j: CR CC
        :param features: SINGULAR
        :return:
        """

        cdef DTYPE_t8 crs = pair_i.shape[0]
        cdef DTYPE_t8 ccs = pair_i.shape[1]
        cdef DTYPE_t8 cr, cc
        cdef DTYPE_t8 i = 0
        cdef DTYPE_t8 j = 0

        # n is the size of cell
        cdef DTYPE_ft32 n = crs * ccs

        cdef np.ndarray[DTYPE_t8, ndim=2] glcm = self.glcm
        glcm[:] = 0

        cdef DTYPE_ft32 mean_i = 0
        cdef DTYPE_ft32 mean_j = 0
        cdef DTYPE_ft32 var_i = 0
        cdef DTYPE_ft32 var_j = 0
        cdef DTYPE_ft32 std = 0

        for cr in range(crs):
            for cc in range(ccs):
                i = pair_i[cr, cc]
                j = pair_j[cr, cc]
                features[CONTRAST] += ((i - j) ** 2)
                mean_i += i
                mean_j += j
                glcm[i, j] += 1
                glcm[j, i] += 1  # Symmetric for ASM.

        # /= n because we summed only
        mean_i /= n
        mean_j /= n

        # MEAN is the average of i, j
        features[MEAN] += (mean_i + mean_j) / 2

        for cr in range(crs):
            for cc in range(ccs):
                i = pair_i[cr, cc]
                j = pair_j[cr, cc]
                features[ASM] += glcm[cr, cc] ** 2
                var_i += (i - mean_i) ** 2
                var_j += (j - mean_j) ** 2

        var_i /= n
        var_j /= n

        features[VAR] += (var_i + var_j) / 2

        # Preemptive auxiliary value
        std = sqrt(var_i) * sqrt(var_j)

        for cr in range(crs):
            for cc in range(ccs):
                i = pair_i[cr, cc]
                j = pair_j[cr, cc]

                if std != 0.0:  # Will explode on 0.0
                    features[CORRELATION] += (i - mean_i) * (j - mean_j) / std

        # Note that because it's a probability, this needs to be /= n
        features[CONTRAST]    /= n
        features[CORRELATION] /= n
        # ASM requires /= n twice as its power wraps its probability.
        # It needs to be divided because it's using a symmetric GLCM.
        features[ASM]         /= (n / 2) ** 2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _binarize(self, np.ndarray[DTYPE_ft32, ndim=3] ar) -> np.ndarray:
        """ This binarizes the 2D image by its min-max """
        return (((ar - ar.min()) / ar.max()) * (self.bins - 1)).astype(np.uint8)

    @cython.boundscheck(False)
    def _pair(self, np.ndarray[DTYPE_t8, ndim=2] ar):

        ar_w = view_as_windows(ar, (self.diameter, self.diameter))
        pairs = []
        if "H" in self.pairs:  pairs.append((ar_w[:-1, :-1], ar_w[:-1, 1:]))
        if "V" in self.pairs:  pairs.append((ar_w[:-1, :-1], ar_w[1:, :-1]))
        if "SE" in self.pairs: pairs.append((ar_w[:-1, :-1], ar_w[1:, 1:]))
        if "NE" in self.pairs: pairs.append((ar_w[1:, :-1], ar_w[1:, 1:]))
        return pairs
