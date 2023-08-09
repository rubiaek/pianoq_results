import matplotlib.pyplot as plt
import numpy as np
import glob
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

        self.optimization = SLMOptimizationResult(glob.glob(f'{self.dir_path}\\*.optimizer2')[0])

    def show(self):
        fig, axes = plt.subplots(3, 2)
        imm = axes[0, 0].imshow(self.diode_before.image)
        fig.colorbar(imm, ax=axes[0, 0])
        axes[0, 0].set_title('diode before')
        imm = axes[1, 0].imshow(self.diode_speckles.image)
        fig.colorbar(imm, ax=axes[1, 0])
        axes[1, 0].set_title('diode speckles')
        imm = axes[2, 0].imshow(self.diode_optimized.image)
        fig.colorbar(imm, ax=axes[2, 0])
        axes[2, 0].set_title('diode optimized')

        imm = axes[0, 1].imshow(self.SPDC_before.real_coins)
        fig.colorbar(imm, ax=axes[0, 1])
        axes[0, 1].set_title('SPDC before')
        imm = axes[1, 1].imshow(self.SPDC_speckles.real_coins)
        fig.colorbar(imm, ax=axes[1, 1])
        axes[1, 1].set_title('SPDC speckles')
        imm = axes[2, 1].imshow(self.SPDC_optimized.real_coins)
        fig.colorbar(imm, ax=axes[2, 1])
        axes[2, 1].set_title('SPDC optimized')
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(self.optimization.costs)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('cost')
        fig.show()
