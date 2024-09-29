import traceback
import pyperclip
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit


class FITSImage(object):
    def __init__(self, path=None):
        self.image = None
        self.path = path
        self.exposure_time = None
        self.pix_size = None
        self.timestamp = None
        self.f = None
        self.header = None

        if path is not None:
            self.loadfrom(path)

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        if path == 0:
            path = pyperclip.paste().strip('"')

        self.f = fits.open(path)
        self.path = path
        self.image = self.f[0].data
        self.header = self.f[0].header
        self.exposure_time = self.header['EXPTIME']
        self.timestamp = self.header.get('DATE-OBS', None)
        self.pix_size = self.header['XPIXSZ'] * 1e-6 * self.header.get('XBINNING', 1)
        self.f.close()

    def show_image(self, aspect=None, title=None):
        # TODO: set the extent so the scale will be in mm
        fig, axes = plt.subplots()
        y, x = self.image.shape
        xmax = (x/2)*self.pix_size
        ymax = (y/2)*self.pix_size
        extent = (-xmax, xmax, -ymax, ymax)
        im = axes.imshow(self.image, aspect=aspect, extent=extent)
        if title:
            axes.set_title(title)
        fig.colorbar(im, ax=axes)
        fig.show()
        return fig, axes

    def _get_slice(self, line_no=None, col_no=None):
        if line_no is not None:
            V = self.image[line_no, :]
            VV = (V - V.min()) / V.max()
        else:
            V = self.image[:, col_no]
            VV = (V - V.min()) / V.max()
        return VV

    def show_slice(self, line_no=None, col_no=None):
        # todo: scale like I do in show_image
        VV = self._get_slice(line_no, col_no)
        fig, ax = plt.subplots()
        ax.plot(VV)
        fig.show()
        return VV

    def fit_to_gaus(self, x00, sig0, line_no=None, col_no=None):
        VV = self._get_slice(line_no, col_no)
        dummy_x = np.arange(len(VV))
        gaus = lambda x, x0, sig: np.exp(-2*((x - x0) ** 2) / (sig ** 2))  # See wiki on Gausian beam. E(x) is with no 2.
        popt, pcov = curve_fit(gaus, dummy_x, VV, p0=(x00, sig0))
        print(popt)

        fig, ax = plt.subplots()
        ax.plot(dummy_x, VV)
        ax.plot(dummy_x, gaus(dummy_x, *popt))
        ax.set_title(f'{line_no=}, {col_no=}')
        fig.show()

        return popt
