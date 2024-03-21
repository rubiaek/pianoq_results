import re
from uncertainties import unumpy
from scipy.signal import convolve2d
import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pianoq_results.klyshko_result import KlyshkoResult, show_memory
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage
from pianoq_results.misc import my_mesh


# PATH_OPTIMIZATION2 = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_24_14_54_48_klyshko_two_diffusers_other_0.25_0.5_power_meter_continuous_hex_in_place'
# PATH_THICK =        r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_28_08_10_25_klyshko_thick_diffuser_0.25_and_0.25_0.16_power_meter_continuous_hex'
# PATH_THICK_MEMORY2 = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_28_08_10_25_klyshko_thick_diffuser_0.25_and_0.25_0.16_power_meter_continuous_hex\memory_measurements'

PATH_OPTIMIZATION = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\2023_09_13_11_19_55_klyshko_very_thick_0.5_and_0.25EDC_0.25EDS'
PATH_THICK_MEMORY = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\2023_09_20_09_52_22_klyshko_very_thick_with_memory_meas\Memory'


def classical_memory(show_ds=(7, 6, 4, 1)):
    dir_path = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Memory\try6\diode_memory'
    paths = glob.glob(f'{dir_path}\\*d=*.fits')
    all_ds = np.array([re.findall('.*d=(.*).fits', path)[0] for path in paths]).astype(float)

    ind = np.where(all_ds == show_ds[0])[0][0]
    im = FITSImage(paths[ind])
    image = im.image
    image -= image.min()
    I_max = image.max()

    fig, axes = plt.subplots(1, len(show_ds), figsize=(len(show_ds)*3, 2.4), constrained_layout=True)
    for i in range(len(show_ds)):
        ind = np.where(all_ds == show_ds[i])[0][0]
        im = FITSImage(paths[ind])
        image = im.image
        image -= image.min()
        imm = axes[i].imshow(image) # , vmax=I_max)
        # ind_row, ind_col = np.unravel_index(np.argmax(image, axis=None), image.shape)
        ind_row, ind_col = 400, 400
        X_pixs = 150
        Y_pixs = 150
        axes[i].set_xlim(left=ind_col - X_pixs / 2, right=ind_col + X_pixs / 2)
        axes[i].set_ylim(bottom=ind_row - Y_pixs / 2, top=ind_row + Y_pixs / 2)
        # if i != 0:
        # if True:
            # axes[i].tick_params(axis='both', which='both', left=False, bottom=True, labelleft=False, labelbottom=True)
        axes[i].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
        # if i == len(show_ds) - 1:
        fig.colorbar(imm, ax=axes[i])

        # axes[i].set_title(f'd = {show_ds[i]}')

    fig.show()
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    fig.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_classical_memory.svg', dpi=fig.dpi)


def memory_old():
    show_memory(PATH_THICK_MEMORY, show_ds=(50, 60, 70, 80, 90, 100, 150, 200), classic=True)
    show_memory(PATH_THICK_MEMORY, show_ds=(50, 60, 70, 80, 90, 100), classic=False)


def optimization():
    res = KlyshkoResult()
    res.loadfrom(PATH_OPTIMIZATION)
    res.print()
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    save_to = rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_optimization.svg'
    res.show(xs=True, bare=True, save_to=save_to, norm_diode=True)
    # res.show_optimization_process()
    # res.show_best_phase()


def similar_speckles():
    DIODE_PATH1 = r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\2023_08_21_10_19_52_diode_speckles.fits"
    SPDC_PATH1 = r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\2023_08_21_11_27_52_scan_trying_Klyshko_1_diffuser_slm_optimized.scan"

    print('one 0.25 deg diffuser I think')
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(7.5, 2.5))
    diode = FITSImage(DIODE_PATH1)
    imm = axes[0].imshow(diode.image)
    axes[0].set_xlim(left=170, right=420)
    axes[0].set_ylim(top=120, bottom=370)

    fig.colorbar(imm, ax=axes[0])
    SPDC = ScanResult(SPDC_PATH1)
    my_mesh(SPDC.X, SPDC.Y, SPDC.real_coins, axes[1])
    axes[1].invert_xaxis()
    fig.show()


def similar_speckles2():
    DIODE_PATH1 = r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\try3_good\2023_12_20_11_52_31_speckles4.fits"
    SPDC_PATH1 = r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\try3_good\2023_12_20_13_11_18_speckles4_two_photon_high_res_again.scan"
    POWER_METER_PATH = r"G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Speckles same\try3_good\2023_12_20_12_00_55_speckles4_power_meter.scan"

    print('one 0.25 deg diffuser I think')
    diode = FITSImage(DIODE_PATH1)
    SPDC = ScanResult(SPDC_PATH1)
    PM = ScanResult(POWER_METER_PATH)

    fig1, ax = plt.subplots()
    imm = ax.imshow(np.fliplr(np.rot90(diode.image)))
    pixels = (SPDC.X[-1] - SPDC.X[0])*1e-3 // diode.pix_size
    start_x = 220
    start_y = 195
    ax.set_xlim(left=start_x, right=start_x+pixels)
    ax.set_ylim(top=start_y, bottom=start_y+pixels)
    ax.axis('off')
    # fig1.colorbar(imm, ax=ax)
    fig1.show()

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    fig1.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_same_speckles_diode_cam.svg', dpi=fig1.dpi)

    fig2, ax = plt.subplots()
    ax.imshow(SPDC.real_coins)
    # my_mesh(SPDC.X, SPDC.Y, SPDC.real_coins, ax)
    ax.invert_xaxis()
    ax.axis('off')
    fig2.show()
    fig2.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_same_speckles_SPDC.svg',
                 dpi=fig2.dpi)

    fig3, ax = plt.subplots()
    ax.imshow(PM.real_coins)
    ax.invert_xaxis()
    ax.axis('off')
    fig3.show()
    fig3.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_same_diode_power_meter.svg',
                 dpi=fig3.dpi)

    # fig.savefig()


def two_spots(norm_diode=True):
    dir_path = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Two spots\Try2_good\try2'
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(7, 4))
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

    classical_speckles = im_speckles.image[mask].astype(float)
    classical_fixed = im_fixed.image[mask].astype(float)

    if norm_diode:
        mean_diode = classical_speckles.mean()
        classical_speckles /= mean_diode
        classical_fixed /= mean_diode

    imm = axes[0, 0].imshow(classical_speckles)
    fig.colorbar(imm, ax=axes[0, 0])
    imm = axes[1, 0].imshow(classical_fixed)
    fig.colorbar(imm, ax=axes[1, 0])
    imm = axes[0, 1].imshow(scan_speckles.real_coins.T)
    fig.colorbar(imm, ax=axes[0, 1])
    imm = axes[1, 1].imshow(scan_fixed.real_coins.T)
    fig.colorbar(imm, ax=axes[1, 1])

    axes[0, 0].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[1, 0].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[0, 1].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[1, 1].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    fig.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_two_spots.svg',dpi=fig.dpi)
    fig.show()


def get_memory_classical3(dir_path, l=4):
    paths = glob.glob(f'{dir_path}\\*d=*.fits')
    dark_im = FITSImage(r"G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try6\2024_01_03_11_26_52_dark.fits")
    all_ds = np.array([re.findall('.*d=(.*)\.fits', path)[0] for path in paths]).astype(float)
    all_ds, paths = list(zip(*sorted(zip(all_ds, paths), key=lambda pair: pair[0], reverse=True)))
    all_ds = np.array(all_ds, dtype=float)
    all_ds *= 10
    ims = [FITSImage(path) for path in paths]
    max_speckles = []

    for i, im in enumerate(ims):
        image = im.image.astype(float) - dark_im.image.astype(float)
        image[image < 0] = 0
        # TODO: have some moving window around expected focus that you choose the max pixel from within it, similar to SPDC
        # TODO: though there is no real need - it happens automatically
        ind_row, ind_col = np.unravel_index(np.argmax(image, axis=None), image.shape)
        # altogether 9 pixels, which comes out ~101um
        l = 4
        max_speckle = image[ind_row - l:ind_row + l + 1, ind_col - l:ind_col + l + 1].sum()
        # in SPDC we have 50um fibers, and counting 3 pixels that are 25um apart, so we count the inner 50um twice
        # 5 pixels is ~56um
        l = 2
        inner_50_um = image[ind_row - l:ind_row + l + 1, ind_col - l:ind_col + l + 1].sum()
        max_speckles.append(max_speckle+inner_50_um)

    dx = all_ds[0] - all_ds

    return dx, np.array(max_speckles) / max_speckles[0]


def get_memory_SPDC3(dir_path, l=1):
    paths = glob.glob(f'{dir_path}\\*d=*.scan')
    all_ds = np.array([re.findall('.*d=(.*)\.scan', path)[0] for path in paths]).astype(float)
    all_ds, paths = list(zip(*sorted(zip(all_ds, paths), key=lambda pair: pair[0], reverse=True)))
    all_ds = np.array(all_ds, dtype=float)
    all_ds *= 10  # the filename says 6 and 7, which is in practice 60 and 70 um
    scans = [ScanResult(path) for path in paths]
    max_speckles = []
    max_speckle_stds = []
    for i, scan in enumerate(scans):
        ind_row, ind_col = np.unravel_index(np.argmax(scan.real_coins, axis=None), scan.real_coins.shape)
        # altogether 3X3 pixels, which comes out ~75umX75um
        coin_area = scan.real_coins[ind_row-l:ind_row+l+1, ind_col-l:ind_col+l+1]
        coin_std_area = scan.real_coins_std[ind_row-l:ind_row+l+1, ind_col-l:ind_col+l+1]
        u_max_speckle = unumpy.uarray(coin_area, coin_std_area).sum()
        max_speckle = u_max_speckle.nominal_value
        max_speckle_std = u_max_speckle.std_dev
        max_speckles.append(max_speckle)
        max_speckle_stds.append(max_speckle_std)

    return all_ds[0] - all_ds, np.array(max_speckles) / max_speckles[0], np.array(max_speckle_stds, dtype=float) / max_speckles[0]


def mem_func(theta, d_theta):
    return ( (theta/d_theta) / (1e-17 + np.sinh(theta/d_theta)) )**2


def show_memories3(dir_path_classical, dir_path_SPDC, d_x=22, l1=3, l2=1, show_fit=True):
    from matplotlib.legend_handler import HandlerTuple

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    diode_ds, diode_corrs = get_memory_classical3(dir_path_classical, l=l1)
    SPDC_ds, SPDC_corrs, SPDC_corr_stds = get_memory_SPDC3(dir_path_SPDC, l=l2)

    diode_thetas = diode_ds * 10 / 100e3  # 10x magnification to SMF, and 100mm lens (e3 for mm instead of um)
    SPDC_thetas = SPDC_ds * 10 / 100e3  # 10x magnification to SMF, and 100mm lens (e3 for mm instead of um)
    theta_err = 2*10/100e3  # 20 um in manual micrometer, and 100mm lens (e3 for mm instead of um)

    spdch = ax.errorbar(SPDC_thetas*1e3, SPDC_corrs, xerr=theta_err*1e3, yerr=SPDC_corr_stds, fmt='o', label='SPDC', color='mediumpurple')
    diodeh = ax.errorbar(diode_thetas*1e3, diode_corrs, xerr=theta_err*1e3, fmt='*', label='diode', color='mediumseagreen')

    if show_fit:
        dummy_theta = np.linspace(1e-6, 0.007, 1000)
        # ax.plot(dummy_x, mem_func(dummy_x, d_x), '-', label='analytical')
        popt, pcov = curve_fit(mem_func, diode_thetas, diode_corrs, p0=0.02, bounds=(1e-6, 2))
        # *1e3 for mrd instead of rad
        fit1h, = ax.plot(dummy_theta*1e3, mem_func(dummy_theta, *popt), '--', label='diode fit', color='mediumseagreen')
        print(f'diode: d_theta= {popt[0]}+-{np.sqrt(pcov[0][0])}')
        popt, pcov = curve_fit(mem_func, SPDC_thetas, SPDC_corrs, p0=0.02, bounds=(1e-6, 2))
        # *1e3 for mrd instead of rad
        fit2h, = ax.plot(dummy_theta*1e3, mem_func(dummy_theta, *popt), '--', label='SPDC fit', color='mediumpurple')
        print(f'SPDC: d_theta= {popt[0]}+-{np.sqrt(pcov[0][0])}')

    reoptimization_x = np.array([7, 5.5, 2, 2])
    reoptimization_x = reoptimization_x[0] - reoptimization_x
    reoptimization_x *= 10 * 10 / 100e3  # 10 for 6->60 um, than 10 for SMF magnification, and 100mm lens (e3 for mm instead of um)

    # reoptimization_y = np.array([120.21360481225001, 101.99433080050001, 37.6670910675, 89.29465100224999])  # smoothen=True
    reoptimization_y = np.array([121.3, 113.2, 44.2, 100.4])  # smoothen = False
    reoptimization_y_errs = np.array([8., 8., 5., 7.])  # smoothen = False

    max_y = reoptimization_y.max()
    reoptimization_y /= max_y
    reoptimization_y_errs /= max_y
    reoptimizationh = ax.errorbar(reoptimization_x*1e3, reoptimization_y,
                                   xerr=theta_err*1e3, yerr=reoptimization_y_errs,
                                   fmt='v', label='reoptimization', color='purple', markersize=8)

    ax.set_xlabel(r'$\Delta\theta$ (mrd)', size=16)
    ax.set_ylabel('normalized focus intensity', size=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if show_fit:
        l = ax.legend([fit1h, fit2h, diodeh, (spdch, reoptimizationh)], ['diode fit', 'SPDC fit', 'diode','SPDC'],
                       handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        l = ax.legend([diodeh, (spdch, reoptimizationh)], ['diode', 'SPDC'],
                       handler_map={tuple: HandlerTuple(ndivide=None)})

    # fig.legend()  # bbox_to_anchor=(0.95, 0.95))
    fig.show()
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    fig.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_memory_curves.svg',
                 dpi=fig.dpi)


def memory(d_x=22, l1=4, l2=1, show_fit=False):
    dir_path_classical = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Memory\try6\diode_memory'
    dir_path_SPDC = r'G:\My Drive\Projects\Klyshko Optimization\Paper1\Data\Memory\try6\SPDC_memory'
    show_memories3(dir_path_classical, dir_path_SPDC, d_x=d_x, l1=l1, l2=l2, show_fit=show_fit)


def reoptimization(smoothen=True, N=4):
    d7 = ScanResult(r"G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try7\2024_01_04_12_14_58_optimized_d=7.scan")
    d5_5 = ScanResult(r"G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try7\2024_01_04_12_51_30_optimized_d=5.5.scan")
    d2 = ScanResult(r"G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try7\2024_01_04_13_19_30_optimized_d=2.scan")
    d2_reoptimized = ScanResult(r"G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try7\2024_01_04_16_19_41_re_optimizedat_d=2_again_d=2.scan")

    fig, axes = plt.subplots(1, 4, figsize=(7, 2), constrained_layout=True)
    dx = (d7.X[1] - d7.X[0]) / 2
    dy = (d7.Y[1] - d7.Y[0]) / 2
    # note the switch here X<->Y because I plot the transpose
    redundant_left = 2
    redundant_right = 6
    extent = (d7.Y[-redundant_left] + dx, d7.Y[0] - dx, d7.X[0] - dy, d7.X[-1] + dy)
    extent=None

    max_V = max(d7.real_coins.max(), d5_5.real_coins.max(), d2.real_coins.max())  # TODO: also of d2_reopt
    max_V = None

    ker = np.ones((2, 2)) / 4
    if smoothen:
        d7coin = convolve2d(d7.real_coins, ker, mode='same')
        d5_5coin = convolve2d(d5_5.real_coins, ker, mode='same')
        d2coin = convolve2d(d2.real_coins, ker, mode='same')
        d2_re_coin = convolve2d(d2_reoptimized.real_coins, ker, mode='same')
    else:
        d7coin = d7.real_coins
        d5_5coin = d5_5.real_coins
        d2coin = d2.real_coins
        d2_re_coin = d2_reoptimized.real_coins

    if False:
        print('sanity check for comparison with and without smoothing:')
        print(f'{d7coin.sum()}')
        print(f'{d5_5coin.sum()}')
        print(f'{d2coin.sum()}')
        print(f'{d2_re_coin.sum()}')
    if False:
        print('check what numbers come good for N highest values')
        print(f'{sum(sorted(d7coin.flatten())[-N:])}')
        print(f'{sum(sorted(d5_5coin.flatten())[-N:])}')
        print(f'{sum(sorted(d2coin.flatten())[-N:])}')
        print(f'{sum(sorted(d2_re_coin.flatten())[-N:])}')
    if True:
        def sum_around_highest(matrix):
            max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
            neighbors_indices = [(i, j) for i in range(max_index[0] - 1, max_index[0] + 2) for j in
                                 range(max_index[1] - 1, max_index[1] + 2) if
                                 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]]
            return np.sum([matrix[i, j] for i, j in neighbors_indices])

        def sum_around_highest2(scan):
            real_coins = scan.real_coins
            real_coin_stds = scan.real_coins_std
            max_index = np.unravel_index(np.argmax(real_coins), real_coins.shape)
            neighbors_indices = [(i, j) for i in range(max_index[0] - 1, max_index[0] + 2) for j in
                                 range(max_index[1] - 1, max_index[1] + 2) if
                                 0 <= i < real_coins.shape[0] and 0 <= j < real_coins.shape[1]]
            u_array = unumpy.uarray(real_coins, real_coin_stds)
            return np.sum([u_array[i, j] for i, j in neighbors_indices])
        print('9 around max')
        # TODO: add error bars (shot noise, with integration_time = 2s)
        print(f'{sum_around_highest2(d7)}')
        print(f'{sum_around_highest2(d5_5)}')
        print(f'{sum_around_highest2(d2)}')
        print(f'{sum_around_highest2(d2_reoptimized)}')

    imm = axes[0].imshow(d7coin[redundant_left:-redundant_right, :], extent=extent, vmax=max_V)
    axes[0].invert_xaxis()
    axes[0].tick_params(axis='y', left=False, labelleft=False)
    axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
    fig.colorbar(imm, ax=axes[0])

    imm = axes[1].imshow(d5_5coin[redundant_left:-redundant_right, :], extent=extent, vmax=max_V)
    axes[1].invert_xaxis()
    axes[1].tick_params(axis='y', left=False, labelleft=False)
    axes[1].tick_params(axis='x', bottom=False, labelbottom=False)
    fig.colorbar(imm, ax=axes[1])

    imm = axes[2].imshow(d2coin[redundant_left:-redundant_right, :], extent=extent, vmax=max_V)
    axes[2].invert_xaxis()
    axes[2].tick_params(axis='y', left=False, labelleft=False)
    axes[2].tick_params(axis='x', bottom=False, labelbottom=False)
    fig.colorbar(imm, ax=axes[2])

    imm = axes[3].imshow(d2_re_coin[redundant_left:-redundant_right, :], extent=extent, vmax=max_V)
    axes[3].invert_xaxis()
    axes[3].tick_params(axis='y', left=False, labelleft=False)
    axes[3].tick_params(axis='x', bottom=False, labelbottom=False)
    fig.colorbar(imm, ax=axes[3])

    fig.show()
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    fig.savefig(rf'G:\My Drive\Projects\Klyshko Optimization\Paper1\Figures\{timestamp}_reoptimization.svg',
                 dpi=fig.dpi)


def print_pump_waist(line_no=1115, col_no=None, x0=None):
    if line_no is not None:
        assert col_no is None
    im = FITSImage(r"G:\My Drive\Projects\Klyshko Optimization\Results\Calibrations\waist_at_crystal.fits")
    sig_pix = im.fit_to_gaus(x0, 200, line_no=line_no, col_no=col_no)[1]
    fig, ax = plt.subplots()
    ax.imshow(im.image)
    if line_no:
        ax.axhline(y=line_no, linestyle='--')
    else:
        ax.axvline(x=col_no, linestyle='--')
    print(f'waist is: {sig_pix*2.2}um')


def main():
    # optimization()
    # thick()
    # memory()
    # similar_speckles2()
    # two_spots()
    # memory(show_fit=True)
    # reoptimization(smoothen=False)
    classical_memory()
    # print_pump_waist(line_no=1115, col_no=None, x0=1500)
    # print_pump_waist(line_no=1045, col_no=None, x0=1500)
    # print_pump_waist(line_no=None, col_no=1530, x0=1100)


if __name__ == '__main__':
    main()
    plt.show()
