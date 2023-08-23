import glob
import json
import SLMlayout
import numpy as np
import matplotlib.pyplot as plt

from pianoq_results.misc import my_mesh
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage
from pianoq_results.slm_optimization_result import SLMOptimizationResult


class KlyshkoResult(object):
    def __init__(self, dir_path=None):
        self.dir_path = dir_path
        self.diode_before = None
        self.diode_speckles = None
        self.diode_optimized = None
        self.SPDC_before = None
        self.SPDC_speckles = None
        self.SPDC_optimized = None
        self.optimization = None

        if self.dir_path:
            self.loadfrom(dir_path)

    def loadfrom(self, dir_path):
        self.dir_path = dir_path
        self.diode_before = FITSImage(glob.glob(f'{self.dir_path}\\*diode_no_diffuser*')[0])
        self.diode_speckles = FITSImage(glob.glob(f'{self.dir_path}\\*diode_speckle*')[0])
        self.diode_optimized = FITSImage(glob.glob(f'{self.dir_path}\\*diode_optimized*')[0])

        self.SPDC_before = ScanResult(glob.glob(f'{self.dir_path}\\*corr_no_diffuser*')[0])
        self.SPDC_speckles = ScanResult(glob.glob(f'{self.dir_path}\\*two_photon_speckle*')[0])
        self.SPDC_optimized = ScanResult(glob.glob(f'{self.dir_path}\\*corr_optimized*')[0])

        self.optimization = SLMOptimizationResult()
        self.optimization.loadfrom(glob.glob(f'{self.dir_path}\\*.optimizer2')[0])

        self.config = json.loads(open(glob.glob(f'{self.dir_path}\\*config.json')[0]).read())
        self.__dict__.update(self.config)
        self.hexs = SLMlayout.Hexagons(radius=self.slm_pinhole_radius, cellSize=self.cell_size,
                                       resolution=(1024, 1272), center=self.slm_pinhole_center, method='equal')

    def show(self, full=False, xs=True):
        fig, axes = plt.subplots(3, 2, figsize=(6.8, 8), constrained_layout=True)
        imm = axes[0, 0].imshow(self.diode_before.image)
        fig.colorbar(imm, ax=axes[0, 0])
        axes[0, 0].set_title('diode before')

        imm = axes[1, 0].imshow(self.diode_speckles.image)
        fig.colorbar(imm, ax=axes[1, 0])
        axes[1, 0].set_title('diode speckles')

        imm = axes[2, 0].imshow(self.diode_optimized.image)
        fig.colorbar(imm, ax=axes[2, 0])
        axes[2, 0].set_title('diode optimized')

        if not full:
            self._set_lims(axes[0, 0])
            self._set_lims(axes[1, 0])
            self._set_lims(axes[2, 0])

        my_mesh(self.SPDC_before.X, self.SPDC_before.Y, self.SPDC_before.real_coins, axes[0, 1])
        axes[0, 1].invert_xaxis()
        axes[0, 1].set_title('SPDC before')

        my_mesh(self.SPDC_speckles.X, self.SPDC_speckles.Y, self.SPDC_speckles.real_coins, axes[1, 1])
        axes[1, 1].invert_xaxis()
        axes[1, 1].set_title('SPDC speckle')

        my_mesh(self.SPDC_optimized.X, self.SPDC_optimized.Y, self.SPDC_optimized.real_coins, axes[2, 1])
        axes[2, 1].invert_xaxis()
        axes[2, 1].set_title('SPDC optimized')

        if xs:
            X_MARKER_COLOR = '#929591'
            X_MARKER_EDGEWITDH = 1.5
            axes[0, 1].plot(self.optimization_x_loc, self.optimization_y_loc, '+', markeredgecolor=X_MARKER_COLOR,
                            markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
            axes[1, 1].plot(self.optimization_x_loc, self.optimization_y_loc, '+', markeredgecolor=X_MARKER_COLOR,
                            markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
            axes[2, 1].plot(self.optimization_x_loc, self.optimization_y_loc, '+', markeredgecolor=X_MARKER_COLOR,
                            markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)

        fig.show()

    def _set_lims(self, ax):
        A = self.diode_before.image
        ind_row, ind_col = np.unravel_index(np.argmax(A, axis=None), A.shape)
        X = self.SPDC_before.X
        Y = self.SPDC_before.Y
        pix_size = self.diode_before.pix_size
        X_pixs = (X[-1] - X[0])*1e-3 / pix_size
        Y_pixs = (Y[-1] - Y[0]) * 1e-3 / pix_size

        ax.set_xlim(left=ind_col - X_pixs/2, right=ind_col + X_pixs/2)
        ax.set_ylim(bottom=ind_row - Y_pixs/2, top=ind_row + Y_pixs/2)

    def show_optimization_process(self):
        fig, ax = plt.subplots()
        ax.plot(self.optimization.costs)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('cost')
        fig.show()

    def show_best_phase(self):
        phase = self.hexs.getImageFromVec(self.optimization.best_phase_mask, dtype=float)
        fig, ax = plt.subplots()
        im = ax.imshow(phase, cmap='gray')
        fig.colorbar(im, ax=ax)
        fig.show()

    @property
    def efficiency_diode(self):
        return self.diode_optimized.image.max() / self.diode_before.image.max()

    @property
    def efficiency_SPDC(self):
        return self.SPDC_optimized.real_coins.max() / self.SPDC_before.real_coins.max()

    @property
    def enhancement_diode(self):
        A = self.diode_before.image
        ind_row, ind_col = np.unravel_index(np.argmax(A, axis=None), A.shape)
        X = self.SPDC_before.X
        Y = self.SPDC_before.Y
        pix_size = self.diode_before.pix_size
        X_pixs = (X.max() - X.min())*1e-3 / pix_size
        Y_pixs = (Y.max() - Y.min()) * 1e-3 / pix_size
        mask_for_enhancement = np.index_exp[int(ind_row - Y_pixs//2): int(ind_row + Y_pixs//2), int(ind_col - X_pixs//2): int(ind_col + X_pixs//2)]
        return self.diode_optimized.image.max() / self.diode_speckles.image[mask_for_enhancement].mean()

    @property
    def enhancement_SPDC(self):
        # TODO: don't take only max in any of the enhancement / efficiency functions, rather somehow the speckle grain
        speckles_coins = self.SPDC_speckles.real_coins.copy()
        speckles_coins[speckles_coins < 0] = 0
        return self.SPDC_optimized.real_coins.max() / speckles_coins.mean()

    def print(self):
        print(f'Diode enhancement: {self.enhancement_diode}')
        print(f'SPDC enhancement: {self.enhancement_SPDC}')
        print(f'Diode efficiency: {self.efficiency_diode}')
        print(f'SPDC efficiency: {self.efficiency_SPDC}')
