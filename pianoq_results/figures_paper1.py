import glob
import json
import matplotlib.pyplot as plt
from pianoq_results.scan_result import ScanResult
from matplotlib_scalebar.scalebar import ScaleBar

COLORMAP = 'viridis'


def add_scalebar(ax):
    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    ax.axis('off')
    ax.add_artist(scalebar)


def show_optimization(dir_path):
    path = glob.glob(f'{dir_path}\\*speckles.scan')[0]
    speckle_scan = ScanResult(path)
    path = glob.glob(f'{dir_path}\\*optimized.scan')[0]
    optimized_scan = ScanResult(path)

    min_coin = min(speckle_scan.real_coins.min(), optimized_scan.real_coins.min())
    max_coin = optimized_scan.real_coins.max()

    max1 = speckle_scan.single2s.max()
    max2 = optimized_scan.single2s.max()
    max_singles = max(max1, max2)

    fig, axes = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)

    im0 = axes[0, 0].imshow(speckle_scan.single2s, vmin=0, vmax=max_singles, cmap=COLORMAP)
    add_scalebar(axes[0, 0])
    # fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[1, 0].imshow(speckle_scan.real_coins, vmin=min_coin, vmax=max_coin, cmap=COLORMAP)
    add_scalebar(axes[1, 0])
    # fig.colorbar(im1, ax=axes[1, 0])

    im2 = axes[0, 1].imshow(optimized_scan.single2s, vmin=0, vmax=max_singles, cmap=COLORMAP)
    add_scalebar(axes[0, 1])
    fig.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 1].imshow(optimized_scan.real_coins, vmin=min_coin, vmax=max_coin, cmap=COLORMAP)
    add_scalebar(axes[1, 1])
    fig.colorbar(im3, ax=axes[1, 1])

    print(f'speckles single1s mean: {speckle_scan.single1s.mean():.0f}, single2s mean: {speckle_scan.single2s.mean():.0f} total coin: {speckle_scan.real_coins.sum():.0f}')
    print(f'optimized single1s mean: {optimized_scan.single1s.mean():.0f}, single2s mean: {optimized_scan.single2s.mean():.0f}, total coin: {optimized_scan.real_coins.sum():.0f}')

    fig.show()


def show_speckles(path1, path2, path3, show_singles=False):
    scan1 = ScanResult(path1)
    scan2 = ScanResult(path2)
    scan3 = ScanResult(path3)

    max_singles = max(scan1.single2s.max(), scan2.single2s.max(), scan3.single2s.max())
    max_coin = max(scan1.real_coins.max(), scan2.real_coins.max(), scan3.real_coins.max())
    min_coin = max(scan1.real_coins.min(), scan2.real_coins.min(), scan3.real_coins.min())

    if show_singles:
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 5), constrained_layout=True)
        im0 = axes[0, 0].imshow(scan1.single2s, vmin=0, vmax=max_singles, cmap=COLORMAP)
        add_scalebar(axes[0, 0])
        # fig.colorbar(im0, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(scan2.single2s, vmin=0, vmax=max_singles, cmap=COLORMAP)
        add_scalebar(axes[0, 1])
        # fig.colorbar(im2, ax=axes[0, 1])

        im2 = axes[0, 2].imshow(scan3.single2s, vmin=0, vmax=max_singles, cmap=COLORMAP)
        add_scalebar(axes[0, 2])
        cbar = fig.colorbar(im2, ax=axes[0, 2])
        cbar.set_label('counts / sec', size=26)

        coin0_ax = axes[1, 0]
        coin1_ax = axes[1, 1]
        coin2_ax = axes[1, 2]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.5), constrained_layout=True)
        coin0_ax, coin1_ax, coin2_ax = axes


    im1 = coin0_ax.imshow(scan1.real_coins, vmin=min_coin, vmax=max_coin, cmap=COLORMAP)
    add_scalebar(coin0_ax)
    # fig.colorbar(im1, ax=axes[1, 0])


    im3 = coin1_ax.imshow(scan2.real_coins, vmin=min_coin, vmax=max_coin, cmap=COLORMAP)
    add_scalebar(coin1_ax)
    # fig.colorbar(im3, ax=axes[1, 1])

    im3 = coin2_ax.imshow(scan3.real_coins, vmin=min_coin, vmax=max_coin, cmap=COLORMAP)
    add_scalebar(coin2_ax)
    cbar = fig.colorbar(im3, ax=coin2_ax)
    cbar.set_label('counts / sec', size=18)
    cbar.ax.tick_params(labelsize=18)

    print(f'scan 1 single1s mean: {scan1.single1s.mean():.0f}, single2s mean: {scan1.single2s.mean():.0f} total coin: {scan1.real_coins.sum():.0f}')
    print(f'scan 2 single1s mean: {scan2.single1s.mean():.0f}, single2s mean: {scan2.single2s.mean():.0f}, total coin: {scan2.real_coins.sum():.0f}')
    print(f'scan 3 single1s mean: {scan3.single1s.mean():.0f}, single2s mean: {scan3.single2s.mean():.0f}, total coin: {scan3.real_coins.sum():.0f}')

    fig.savefig(r'G:\My Drive\Projects\Quantum Piano\Paper 1\Figures\speckles_from_heralded.svg', dpi=fig.dpi)
    fig.show()


def show_two_spots(path_heralded, path_not_heralded, add_circles=True):
    path = glob.glob(f'{path_heralded}\\*optimized.scan')[0]
    h_scan = ScanResult(path)
    path = glob.glob(f'{path_heralded}\\*config.json')[0]
    h_json = json.loads(open(path).read())

    path = glob.glob(f'{path_not_heralded}\\*optimized.scan')[0]
    nh_scan = ScanResult(path)
    path = glob.glob(f'{path_not_heralded}\\*config.json')[0]
    nh_json = json.loads(open(path).read())

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    h_ax, nh_ax = axes

    # Heralded
    im_h = h_ax.imshow(h_scan.real_coins2, cmap=COLORMAP, extent=h_scan.extent)
    add_scalebar(h_ax)
    h_circ1 = plt.Circle(h_json['optimized_xy'], 0.0625, color='b', fill=False)

    other_y = h_json['optimized_xy'][1] - h_scan.displacement_between_coins[0]
    other_x = h_json['optimized_xy'][0] + h_scan.displacement_between_coins[1]
    h_circ2 = plt.Circle((other_x, other_y), 0.0625, color='r', fill=False)

    if add_circles:
        h_ax.add_patch(h_circ1)
        h_ax.add_patch(h_circ2)

    # cbar = fig.colorbar(im_h, ax=h_ax)
    # cbar.set_label('counts / sec', size=18)
    # cbar.ax.tick_params(labelsize=18)

    # Not Heralded
    im_nh = nh_ax.imshow(nh_scan.real_coins2, cmap=COLORMAP, extent=nh_scan.extent)
    add_scalebar(nh_ax)
    nh_circ1 = plt.Circle(nh_json['optimized_xy'], 0.0625, color='b', fill=False)

    other_y = nh_json['optimized_xy'][1] - nh_scan.displacement_between_coins[0]
    other_x = nh_json['optimized_xy'][0] + nh_scan.displacement_between_coins[1]
    nh_circ2 = plt.Circle((other_x, other_y), 0.0625, color='r', fill=False)

    if add_circles:
        nh_ax.add_patch(nh_circ1)
        nh_ax.add_patch(nh_circ2)

    cbar = fig.colorbar(im_nh, ax=nh_ax)
    cbar.set_label('counts / sec', size=18)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(r'G:\My Drive\Projects\Quantum Piano\Paper 1\Figures\two_spots.svg', dpi=fig.dpi)
    fig.show()


############### optimization ###############
path_not_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\2022_12_27_15_52_37_for_optimization_integration_8s_all'
path_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\2022_12_19_02_50_01_optimization_integration_5s_all_same'
# show_optimization(path_heralded)
# show_optimization(path_not_heralded)

print()

############### speckles ###############
NOT_HERALDED_PATH_FORMAT = r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\Many speckles\2022_12_28_15_50_55_{num}_speckles.scan"

# show_speckles(NOT_HERALDED_PATH_FORMAT.format(num=6),
#               NOT_HERALDED_PATH_FORMAT.format(num=7),
#               NOT_HERALDED_PATH_FORMAT.format(num=5))

print()

HERALDED_PATH_FORMAT = r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\Many Speckles\2023_01_01_11_36_57_{num}_speckles.scan"
# show_speckles(HERALDED_PATH_FORMAT.format(num=1),
#               HERALDED_PATH_FORMAT.format(num=3),
#               HERALDED_PATH_FORMAT.format(num=5), show_singles=False)

############### two spots ###############
TWO_SPOTS_HEREALDED_PATH = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Two Spots\Heralded\2023_01_04_20_15_36_best_double_spot_2'
TWO_SPOTS_NOT_HEREALDED_PATH = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Two Spots\Not Heralded\2023_01_08_11_51_22_double_spot_integration_2s_pretty_good'
show_two_spots(TWO_SPOTS_HEREALDED_PATH, TWO_SPOTS_NOT_HEREALDED_PATH)