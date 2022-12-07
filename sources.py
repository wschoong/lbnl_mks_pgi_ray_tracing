import numpy as np


class Sources(object):
    def __init__(self, center=(0, 0, 0),  # mm Coordinates of the center of Source Space
                 voxel_size=1.0,  # in mm, this is pixel size for 2D
                 npix_1=200,
                 npix_2=200,
                 sax_1=(1, 0, 0),  # (s)ource (ax)is
                 sax_2=(0, 1, 0),
                 prepend_n_ax1=0,
                 prepend_n_ax2=0  # In the negative direction, this means down and to the left
                 ):
        self.sc = np.array(center)
        self.vsze = voxel_size
        self.npix = np.array((npix_1, npix_2))  # np.array((npix_1, npix_2, npix_3))

        self.s_ax = np.array([sax_1, sax_2])  # np.array([sax_1, sax_2, sax_3])
        # self.sax3 = sax_3
        self.prepend = np.array([prepend_n_ax1, prepend_n_ax2])

    def source_pt_iterator(self):
        # For me this meant starting from upper left (facing collimator) and going right then down each row
        for ax1 in (np.arange((-self.npix[1] / 2. + 0.5),
                              (self.npix[1] / 2. + 0.5))[::-1] - self.prepend[1]) * self.vsze:
            for ax0 in (np.arange((-self.npix[0] / 2. + 0.5),
                                  (self.npix[0] / 2. + 0.5)) - self.prepend[0]) * self.vsze:
                yield ax0 * self.s_ax[0] + ax1 * self.s_ax[1] + self.sc
