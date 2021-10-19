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
    cdef public DTYPE_t8 radius, bins, diameter, D2, step_size
    cdef public np.ndarray ar
    cdef public np.ndarray features
    cdef public np.ndarray glcm
    cdef public tuple pairs
    cdef public float CORR_ERROR_VAL
    cdef public char verbose

    def __init__(self, np.ndarray[DTYPE_ft32, ndim=3] ar,
                 DTYPE_t8 radius, DTYPE_t8 bins,
                 verbose=True,
                 step_size=1,
                 pairs=('N', 'W', 'NW', 'SW')):
        self.radius = radius
        self.step_size = step_size
        self.diameter = radius * 2 + 1
        self.bins = bins
        self.ar = ar
        self.CORR_ERROR_VAL = -2

        # Dimensions of the features are
        # ROW, COL, CHN, GLCM_FEATURE
        self.features = np.zeros([<int>(ar.shape[0] - self.diameter - 1 - (<int> step_size - 1) * 2),
                                  <int>(ar.shape[1] - self.diameter - 1 - (<int> step_size - 1) * 2),
                                  ar.shape[2], 5],
                                 dtype=np.float32)
        self.glcm = np.zeros([bins, bins], dtype=np.float32)
        self.pairs = pairs
        self.verbose = verbose

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_glcm(self):
        # This creates the mem_views
        cdef np.ndarray[DTYPE_ft32, ndim=3] ar = self.ar
        cdef np.ndarray[DTYPE_ft32, ndim=4] features = self.features

        # With an input of the ar(float), it binarizes and outputs to ar_bin
        cdef np.ndarray[DTYPE_t8, ndim=3] ar_bin = self._binarize(ar)

        # This is the number of channels of the array
        # E.g. if RGB, then 3.
        cdef DTYPE_t8 chs = <DTYPE_t8>ar_bin.shape[2]

        # This initializes the progress bar wrapper
        with tqdm(total=chs * len(self.pairs), disable=not self.verbose,
                  desc=f"GLCM Progress") as pbar:

            # We can treat each channel independently of the GLCM calculation
            for ch in range(chs):

                # directions is
                # List of Directions
                # Each Direction is a tuple of original windows and offset windows
                # directions: List[Tuple[List[window_i], List[window_j]]
                directions = self._paired_windows(ar_bin[..., ch])

                for direction in directions:
                    # Each pair is a tuple
                    # Tuple of 2 offset images for GLCM calculation.
                    self._populate_glcm(direction[0], direction[1], features[:,:,ch,:])
                    pbar.update()

        # The following statements will rescale the features to [0,1]
        # To fully understand why I do this, refer to my research journal.
        features[..., CONTRAST]    /= (self.bins - 1) ** 2
        features[..., MEAN]        /= self.bins - 1
        features[..., VAR]         /= (self.bins - 1) ** 2
        features[..., CORRELATION] = (features[..., CORRELATION] + len(self.pairs)) / 2

        return self.features / len(self.pairs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _populate_glcm(self,
                       np.ndarray[DTYPE_t8, ndim=4] windows_i,
                       np.ndarray[DTYPE_t8, ndim=4] windows_j,
                       np.ndarray[DTYPE_ft32, ndim=3] features):
        """ For each window pair, this populates a full GLCM.

        The ar would be WR, WC, CR, CC

        :param windows_i: WR WC CR CC
        :param windows_j: WR WC CR CC
        :return:
        """
        cdef DTYPE_t16 wrs = <DTYPE_t16>windows_i.shape[0]
        cdef DTYPE_t16 wcs = <DTYPE_t16>windows_i.shape[1]
        cdef DTYPE_t16 wr = 0;
        cdef DTYPE_t16 wc = 0;

        for wr in range(wrs):
            for wc in range(wcs):
                # For each window_i = windows_i[wr, wc], window_j = windows_j[wr, wc]
                # We want to create the glcm and put into features[wr, wc]
                self._populate_glcm_single(windows_i[wr, wc], windows_j[wr, wc], features[wr, wc])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _populate_glcm_single(self,
                              np.ndarray[DTYPE_t8, ndim=2] window_i,
                              np.ndarray[DTYPE_t8, ndim=2] window_j,
                              np.ndarray[DTYPE_ft32, ndim=1] features):
        """

        :param window_i: CR CC
        :param window_j: CR CC
        :param features: SINGULAR
        :return:
        """

        cdef DTYPE_t8 cr, cc
        cdef DTYPE_t8 i = 0
        cdef DTYPE_t8 j = 0


        # This is the maximum value of G_ij possible
        # This is multiplied by 2 because we're adding its transpose,
        # for bi-directionality.
        
        cdef np.ndarray[DTYPE_ft32, ndim=2] glcm = self.glcm
        glcm[:] = 0

        cdef DTYPE_ft32 mean_i = 0
        cdef DTYPE_ft32 mean_j = 0
        cdef DTYPE_ft32 var_i = 0
        cdef DTYPE_ft32 var_j = 0
        cdef DTYPE_ft32 std = 0
        cdef DTYPE_ft32 corr_num = 0
        cdef DTYPE_ft32 corr_den = 0


        # We populate the GLCM here
        for cr in range(self.diameter):
            for cc in range(self.diameter):
                i = window_i[cr, cc]
                j = window_j[cr, cc]
                mean_i += i
                mean_j += j
                glcm[i, j] += <DTYPE_ft32> (1 / (2 * <DTYPE_ft32>(self.diameter ** 2)))
                glcm[j, i] += <DTYPE_ft32> (1 / (2 * <DTYPE_ft32>(self.diameter ** 2))) # Symmetric for ASM.

        mean_i /= self.diameter ** 2
        mean_j /= self.diameter ** 2

        # For each cell in the GLCM

        for cr in range(self.bins):
            for cc in range(self.bins):
                features[CONTRAST] += glcm[cr, cc] * ((i - j) ** 2)
                features[ASM] += glcm[cr, cc] ** 2
                var_i += glcm[cr, cc] * (cr - mean_i) ** 2
                var_j += glcm[cr, cc] * (cc - mean_j) ** 2

        std = <DTYPE_ft32> (sqrt(var_i) * sqrt(var_j))

        if std != 0:
            for cr in range(self.bins):
                for cc in range(self.bins):
                    features[CORRELATION] += glcm[cr, cc] * (cr - mean_i) * (cc - mean_j) / std

        features[MEAN] += <DTYPE_ft32> ((mean_i + mean_j) / 2)
        features[VAR] += <DTYPE_ft32> ((var_i + var_j) / 2)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _binarize(self, np.ndarray[DTYPE_ft32, ndim=3] ar) -> np.ndarray:
        """ This binarizes the 2D image by its min-max """
        if ar.max() != 0:
            return ((ar / ar.max()) * (self.bins - 1)).astype(np.uint8)
        else:
            return ar.astype(np.uint8)

    @cython.boundscheck(False)
    def _paired_windows(self, np.ndarray[DTYPE_t8, ndim=2] ar):
        """ Creates the pair wise windows.

        Note that this has an offset issue for corner values.
        Thus it'll cropping out the external bounds.

        E.g.
        +---------+  Here, O has a lack of 5 directions, considering 8 directions
        |O        |  Thus, it'll not be considered
        |         |
        |         |
        +---------+

        +---------+  It's trivial to see why the following locations are the only
        |         |  Places where it's possible to use all 8 directions.
        | OOOOOOO |
        |         |  Even though the user may specify a set of directions that may
        +---------+  allow a larger valid bound, we will assume this scenario to
        simplify implementation!

        The algorithm here creates pair-wise by simply having the original window and
        the offset window

        +-----+          +-----+   |   +-----+          +-----+   |
        |     |          |OOO  |   |   |     |          | OOO |   |
        | OOO |   NWest  |OOO  |   |   | OOO |   North  | OOO |   |
        | OOO | <-Pair-> |OOO  |   |   | OOO | <-Pair-> | OOO |   |
        | OOO |          |     |   |   | OOO |          |     |   |
        |     |          |     |   |   |     |          |     |   |
        +-----+          +-----+   |   +-----+          +-----+   |
                                   |                              |
        ---------------------------+------------------------------+
                                   |
         +-----+          +-----+  |  Notice that if did ALL 8 directions,
         |     |          |     |  |  you will find that the algorithm will
         | OOO |   West   |OOO  |  |  have redundancy. This is the reason
         | OOO | <-Pair-> |OOO  |  |  why NE, E, SE, S are missing.
         | OOO |          |OOO  |  |
         |     |          |     |  |  Take for example
         +-----+          +-----+  |  +-----+ Point O calculates GLCM on all
                                   |  |GGG  | 8 Directions, G
        ---------------------------+  |GO#  |
                                   |  |GGG  | In particular, (O, #) will be redundant
         +-----+          +-----+  |  +-----+
         |     |          |     |  |  +-----+ Moving Point O to the right,
         | OOO |   SWest  |     |  |  | GGG |
         | OOO | <-Pair-> |OOO  |  |  | #OG | Notice that (O, #) is the same
         | OOO |          |OOO  |  |  | GGG | GLCM pair as previously.
         |     |          |OOO  |  |  +-----+
         +-----+          +-----+  |          This is because the GLCM is bi-directional

        This redundancy is accounted for in the formula below.

        So "S" is also interpreted as "N". "SE" as "NW" and so on.
        This doesn't cause duplicates.

        :param ar: The binarized array
        :return: Returns a List of Tuple[Pair A, Pair B]
        """

        ar_w = view_as_windows(ar, (self.diameter, self.diameter))
        pairs = []
        cdef DTYPE_t8 s = self.step_size
        original = ar_w[s:-s, s:-s]

        if ("N"  in self.pairs) or ("S"  in self.pairs):
            pairs.append((original, ar_w[:-1-s, s:-s]))
        if ("W"  in self.pairs) or ("E"  in self.pairs):
            pairs.append((original, ar_w[s:-s, 1+s:]))
        if ("NW" in self.pairs) or ("SE" in self.pairs):
            pairs.append((original, ar_w[:-1-s, 1+s:]))
        if ("SW" in self.pairs) or ("NE" in self.pairs):
            pairs.append((original, ar_w[1+s:, 1+s:]))

        return pairs
