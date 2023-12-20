import numpy as np
import matplotlib.pyplot as plt
from pianoq_results.klyshko_result import KlyshkoResult, show_memory
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage
from pianoq_results.misc import my_mesh

# PATH_OPTIMIZATION2 = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_24_14_54_48_klyshko_two_diffusers_other_0.25_0.5_power_meter_continuous_hex_in_place'
# PATH_THICK =        r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_28_08_10_25_klyshko_thick_diffuser_0.25_and_0.25_0.16_power_meter_continuous_hex'
# PATH_THICK_MEMORY2 = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_28_08_10_25_klyshko_thick_diffuser_0.25_and_0.25_0.16_power_meter_continuous_hex\memory_measurements'

PATH_OPTIMIZATION = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\2023_09_13_11_19_55_klyshko_very_thick_0.5_and_0.25EDC_0.25EDS'
PATH_THICK_MEMORY = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\2023_09_20_09_52_22_klyshko_very_thick_with_memory_meas\Memory'


def memory():
    show_memory(PATH_THICK_MEMORY, show_ds=(50, 60, 70, 80, 90, 100, 150, 200), classic=True)
    show_memory(PATH_THICK_MEMORY, show_ds=(50, 60, 70, 80, 90, 100), classic=False)


def optimization():
    res = KlyshkoResult()
    res.loadfrom(PATH_OPTIMIZATION)
    res.print()
    res.show()
    # res.show_optimization_process()
    # res.show_best_phase()


def similar_speckles():
    print('one 0.25 deg diffuser I think')
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(7.5, 2.5))
    diode = FITSImage(r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\2023_08_21_10_19_52_diode_speckles.fits")
    imm = axes[0].imshow(diode.image)
    axes[0].set_xlim(left=170, right=420)
    axes[0].set_ylim(top=120, bottom=370)

    fig.colorbar(imm, ax=axes[0])
    SPDC = ScanResult(r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\2023_08_21_11_27_52_scan_trying_Klyshko_1_diffuser_slm_optimized.scan")
    my_mesh(SPDC.X, SPDC.Y, SPDC.real_coins, axes[1])
    axes[1].invert_xaxis()
    fig.show()


def two_spots():
    dir_path = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Two spots\Try2_good\try2'
    fig, axes = plt.subplots(2, 2)
    im_speckles = FITSImage(dir_path + '\\2023_12_18_11_21_56_speckles.fits')
    im_fixed = FITSImage(dir_path + '\\2023_12_18_14_22_57_optimized_two_spots_cell_size=20.fits')
    scan_speckles = ScanResult(dir_path + '\\2023_12_18_15_14_31_speckles_also_9s_integration.scan')
    scan_fixed = ScanResult(dir_path + '\\2023_12_18_14_26_59_optimized_two_spots_longer_again_pretty_good.scan')

    Dx = np.abs(scan_speckles.X[-1] - scan_speckles.X[0]) * 1e-3  # m
    Dy = np.abs(scan_speckles.Y[-1] - scan_speckles.Y[0]) * 1e-3  # m
    pixs_X = Dx / im_fixed.pix_size
    pixs_Y = Dy / im_fixed.pix_size

    # 255, 250 is magic, this is the middle between both spots. Can probably find it using center of mass or something
    mask = np.index_exp[int(255 - pixs_X//2) : int(255 + pixs_X//2), int(250 - pixs_Y//2): int(250 + pixs_Y//2)]

    imm = axes[0, 0].imshow(im_speckles.image[mask])
    fig.colorbar(imm, ax=axes[0, 0])
    imm = axes[1, 0].imshow(im_fixed.image[mask])
    fig.colorbar(imm, ax=axes[1, 0])
    imm = axes[0, 1].imshow(scan_speckles.real_coins.T)
    fig.colorbar(imm, ax=axes[0, 1])
    imm = axes[1, 1].imshow(scan_fixed.real_coins.T)
    fig.colorbar(imm, ax=axes[1, 1])
    fig.show()


def main():
    # optimization()
    # thick()
    # memory()
    # similar_speckles()
    two_spots()


if __name__ == '__main__':
    main()
    plt.show()
