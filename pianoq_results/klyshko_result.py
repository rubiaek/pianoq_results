import re
import glob
import json
import SLMlayout
import numpy as np
import matplotlib.pyplot as plt

from pianoq_results.misc import my_mesh
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage
from pianoq_results.slm_optimization_result import SLMOptimizationResult
from uncertainties import unumpy


class KlyshkoResult(object):
    def __init__(self, dir_path=None):
        self.dir_path = dir_path
        self.diode_before = None
        self.diode_speckles = None
        self.diode_optimized = None
        self.diode_dark = None
        self.SPDC_before = None
        self.SPDC_speckles = None
        self.SPDC_optimized = None
        self.optimization = None

        if self.dir_path:
            self.loadfrom(dir_path)

    def loadfrom(self, dir_path):
        self.dir_path = dir_path
        self.diode_before = self._load_fits('diode_no_diffuser')
        self.diode_speckles = self._load_fits('diode_speckles')
        self.diode_optimized = self._load_fits('diode_optimized')
        self.diode_dark = self._load_fits('dark')

        self._subtract_dark_from_diode_images()

        self.SPDC_before = self._load_scan('corr_no_diffuser')
        self.SPDC_speckles = self._load_scan('two_photon_speckle')
        self.SPDC_optimized = self._load_scan('corr_optimized')

        self.optimization = SLMOptimizationResult()
        self.optimization.loadfrom(glob.glob(f'{self.dir_path}\\*.optimizer2')[0])

        self.config = json.loads(open(glob.glob(f'{self.dir_path}\\*config.json')[0]).read())
        self.__dict__.update(self.config)
        self.hexs = SLMlayout.Hexagons(radius=self.slm_pinhole_radius, cellSize=self.cell_size,
                                       resolution=(1024, 1272), center=self.slm_pinhole_center, method='equal')

    def _load_fits(self, title):
        paths = glob.glob(f'{self.dir_path}\\*{title}*')
        if len(paths) == 0:
            im = FITSImage()
            im.image = np.zeros((800, 800))
            return im
        elif len(paths) == 1:
            return FITSImage(paths[0])
        elif len(paths) == 2:
            print(f'####### Got two paths: {paths} #########')
            print('Using first one')
            return FITSImage(sorted(paths)[0])
        else:
            raise Exception("Why two paths the same? ")

    def _load_scan(self, title):
        paths = glob.glob(f'{self.dir_path}\\*{title}*')
        if len(paths) == 0:
            scan = ScanResult()
            scan.single1s = np.zeros((20, 20))
            scan.single2s = np.zeros((20, 20))
            scan.coincidences = np.zeros((20, 20))
            return scan
        elif len(paths) == 1:
            return ScanResult(paths[0])
        else:
            raise Exception("Why two paths the same? ")

    def _subtract_dark_from_diode_images(self):
        if self.diode_dark.image.shape != self.diode_before.image.shape:
            print('couldn\'t subtract dark counts from diode images')
            return
        elif self.diode_dark.image.sum() == 0:
            print('no dark image')
            return
        else:
            self.diode_optimized.image = self.diode_optimized.image.astype(float) - self.diode_dark.image
            self.diode_optimized.image[self.diode_optimized.image < 0] = 0

            self.diode_speckles.image = self.diode_speckles.image.astype(float) - self.diode_dark.image
            self.diode_speckles.image[self.diode_speckles.image < 0] = 0

            self.diode_before.image = self.diode_before.image.astype(float) - self.diode_dark.image
            self.diode_before.image[self.diode_before.image < 0] = 0

    def show(self, full=False, xs=True, bare=False, save_to=None, norm_diode=False):
        fig, axes = plt.subplots(3, 2, figsize=(6.8, 8), constrained_layout=True)

        if not full:
            diode_before = self._crop_image(self.diode_before.image)
            diode_speckles = self._crop_image(self.diode_speckles.image)
            diode_optimized = self._crop_image(self.diode_optimized.image)
        else:
            diode_before = self.diode_before.image
            diode_speckles = self.diode_speckles.image
            diode_optimized = self.diode_optimized.image

        if norm_diode:
            mean_diode = diode_speckles.mean()
            diode_before /= mean_diode
            diode_speckles /= mean_diode
            diode_optimized /= mean_diode

        imm = axes[0, 0].imshow(diode_before)
        fig.colorbar(imm, ax=axes[0, 0])

        imm = axes[1, 0].imshow(diode_speckles)
        fig.colorbar(imm, ax=axes[1, 0])

        imm = axes[2, 0].imshow(diode_optimized)
        fig.colorbar(imm, ax=axes[2, 0])

        if not bare:
            axes[0, 0].set_title('diode before')
            axes[1, 0].set_title('diode speckles')
            axes[2, 0].set_title('diode optimized')

        my_mesh(self.SPDC_before.X, self.SPDC_before.Y, self.SPDC_before.real_coins, axes[0, 1])
        axes[0, 1].invert_xaxis()

        my_mesh(self.SPDC_speckles.X, self.SPDC_speckles.Y, self.SPDC_speckles.real_coins, axes[1, 1])
        axes[1, 1].invert_xaxis()

        my_mesh(self.SPDC_optimized.X, self.SPDC_optimized.Y, self.SPDC_optimized.real_coins, axes[2, 1])
        axes[2, 1].invert_xaxis()
        if not bare:
            axes[0, 1].set_title('SPDC before')
            axes[1, 1].set_title('SPDC speckle')
            axes[2, 1].set_title('SPDC optimized')

        if bare:
            axes[0, 0].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            axes[1, 0].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            axes[2, 0].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            axes[0, 1].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            axes[1, 1].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            axes[2, 1].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

        if xs:
            X_MARKER_COLOR = '#929591'
            X_MARKER_EDGEWITDH = 1.5
            axes[0, 1].plot(self.optimization_x_loc, self.optimization_y_loc, '+', markeredgecolor=X_MARKER_COLOR,
                            markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
            axes[1, 1].plot(self.optimization_x_loc, self.optimization_y_loc, '+', markeredgecolor=X_MARKER_COLOR,
                            markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
            axes[2, 1].plot(self.optimization_x_loc, self.optimization_y_loc, '+', markeredgecolor=X_MARKER_COLOR,
                            markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)

        if save_to:
            fig.savefig(save_to, dpi=fig.dpi)
        fig.show()

    def _crop_image(self, im):
        A = self.diode_before.image
        ind_row, ind_col = np.unravel_index(np.argmax(A, axis=None), A.shape)
        X = self.SPDC_before.X
        Y = self.SPDC_before.Y
        pix_size = self.diode_before.pix_size
        X_pixs = np.abs((X[-1] - X[0])*1e-3 / pix_size)
        Y_pixs = np.abs((Y[-1] - Y[0]) * 1e-3 / pix_size)

        return im[int(ind_row - Y_pixs/2): int(ind_row + Y_pixs/2), int(ind_col - X_pixs/2): int(ind_col + X_pixs/2)]

        # ax.set_xlim(left=ind_col - X_pixs/2, right=ind_col + X_pixs/2)
        # ax.set_ylim(bottom=ind_row - Y_pixs/2, top=ind_row + Y_pixs/2)

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
        return phase

    @property
    def efficiency_diode(self):
        cutoff_optimized = self.diode_optimized.image.max() / 2
        optimized_vec = self.diode_optimized.image[self.diode_optimized.image > cutoff_optimized]
        optimized = optimized_vec.sum() / optimized_vec.size

        cutoff_no_diffuser = self.diode_before.image.max() / 2
        no_diffuser_vec = self.diode_before.image[self.diode_before.image > cutoff_no_diffuser]
        no_diffuser = no_diffuser_vec.sum() / no_diffuser_vec.size

        return optimized / no_diffuser

    @property
    def efficiency_SPDC(self):
        optimized = sum_around_highest(self.SPDC_optimized) / 9
        print(f'SPDC_{optimized=}')
        no_diffuser = sum_around_highest(self.SPDC_before) / 9
        print(f'SPDC_{no_diffuser=}')

        return optimized / no_diffuser

    @property
    def enhancement_diode(self):
        im_speckles = self._crop_image(self.diode_speckles.image)

        cutoff_optimized = self.diode_optimized.image.max() / 2
        optimized_vec = self.diode_optimized.image[self.diode_optimized.image > cutoff_optimized]
        optimized = optimized_vec.sum() / optimized_vec.size

        return optimized / im_speckles.mean()

    @property
    def enhancement_SPDC(self):
        optimized = sum_around_highest(self.SPDC_optimized) / 9
        print(f'SPDC_{optimized=}')
        u_array = unumpy.uarray(self.SPDC_speckles.real_coins, self.SPDC_speckles.real_coins_std)
        speckles = u_array.sum() / u_array.size
        print(f'SPDC_{speckles=}')

        # it is OK to sum and then redact accidentals, nothing non-physical here
        # speckles_coins[speckles_coins < 0] = 0
        return optimized / speckles

    def print(self):
        print('#########################')
        print(f'comment: {self.comment}')
        print('#########################')
        print(f'Diode enhancement: {self.enhancement_diode}')
        print(f'SPDC enhancement: {self.enhancement_SPDC}')
        print(f'Diode efficiency: {self.efficiency_diode}')
        print(f'SPDC efficiency: {self.efficiency_SPDC}')

    def reload(self):
        self.loadfrom(self.dir_path)


def show_speckle_comparison(dir_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    # phase mask
    diffuser_path = glob.glob(f'{dir_path}\\*{title}*npz')[0]
    phase_mask = np.load(diffuser_path)['diffuser']
    imm = axes[0].imshow(phase_mask, cmap='gray')
    fig.colorbar(imm, ax=axes[0])
    axes[0].set_title('diffuser phase')

    # Diode
    diode_path = glob.glob(f'{dir_path}\\*{title}*fits')[0]
    diode_im = FITSImage(diode_path)
    imm = axes[1].imshow(diode_im.image)
    fig.colorbar(imm, ax=axes[1])
    axes[1].set_title('diode speckle')

    # SPDC scan
    SPDC_path = glob.glob(f'{dir_path}\\*{title}*scan')[0]
    scan = ScanResult(SPDC_path)
    my_mesh(scan.X, scan.Y, scan.real_coins, axes[2])
    axes[2].invert_xaxis()
    axes[2].set_title('SPDC speckle')

    # diode limits
    diode_before_path = glob.glob(f'{dir_path}\\*normal*fits')[0]
    diode_before_im = FITSImage(diode_path)
    A = diode_before_im.image
    ind_row, ind_col = np.unravel_index(np.argmax(A, axis=None), A.shape)
    X = scan.X
    Y = scan.Y
    pix_size = diode_before_im.pix_size
    X_pixs = (X[-1] - X[0])*1e-3 / pix_size
    Y_pixs = (Y[-1] - Y[0]) * 1e-3 / pix_size
    axes[1].set_xlim(left=ind_col - X_pixs/2, right=ind_col + X_pixs/2)
    axes[1].set_ylim(bottom=ind_row - Y_pixs/2, top=ind_row + Y_pixs/2)

    fig.show()


def show_memory(dir_path, show_ds=(50, 150, 250), classic=False):
    if classic:
        paths = glob.glob(f'{dir_path}\\*d=*um.fits')
    else:
        paths = glob.glob(f'{dir_path}\\*d=*um.scan')
    all_ds = np.array([re.findall('.*d=(.*)um', path)[0] for path in paths]).astype(int)

    fig, axes = plt.subplots(1, len(show_ds), figsize=(len(show_ds)*3.5, 3), constrained_layout=True)
    for i in range(len(show_ds)):
        ind = np.where(all_ds == show_ds[i])[0][0]
        if not classic:
            scan = ScanResult(paths[ind])
            my_mesh(scan.X, scan.Y, scan.real_coins, axes[i])
            axes[i].invert_xaxis()
        else:
            im = FITSImage(paths[ind])
            imm = axes[i].imshow(im.image)
            ind_row, ind_col = np.unravel_index(np.argmax(im.image, axis=None), im.image.shape)
            X_pixs = 266
            Y_pixs = 266
            axes[i].set_xlim(left=ind_col - X_pixs / 2, right=ind_col + X_pixs / 2)
            axes[i].set_ylim(bottom=ind_row - Y_pixs / 2, top=ind_row + Y_pixs / 2)
            fig.colorbar(imm, ax=axes[i])

        axes[i].set_title(f'd = {show_ds[i]}')

    fig.suptitle(f'classic = {classic}')

    fig.show()


def sum_around_highest(scan):
    real_coins = scan.real_coins
    real_coin_stds = scan.real_coins_std
    u_array = unumpy.uarray(real_coins, real_coin_stds)
    max_index = np.unravel_index(np.argmax(real_coins), real_coins.shape)
    neighbors_indices = [(i, j) for i in range(max_index[0] - 1, max_index[0] + 2) for j in
                         range(max_index[1] - 1, max_index[1] + 2) if
                         0 <= i < real_coins.shape[0] and 0 <= j < real_coins.shape[1]]

    return np.sum([u_array[i, j] for i, j in neighbors_indices])


# res = KlyshkoResult(r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\2023_09_13_11_19_55_klyshko_very_thick_0.5_and_0.25EDC_0.25EDS")
# res.print()
