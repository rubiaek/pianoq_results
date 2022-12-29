import glob
import matplotlib.pyplot as plt
from pianoq_results.scan_result import ScanResult
from matplotlib_scalebar.scalebar import ScaleBar


def show_optimization(dir_path):
    path = glob.glob(f'{dir_path}\\*speckles.scan')[0]
    speckle_scan = ScanResult(path)
    path = glob.glob(f'{dir_path}\\*optimized.scan')[0]
    optimized_scan = ScanResult(path)

    min_coin = min(speckle_scan.real_coins.min(), optimized_scan.real_coins.min())
    max_coin = optimized_scan.real_coins.max()
    print(min_coin)

    max1 = speckle_scan.single2s.max()
    max2 = optimized_scan.single2s.max()
    max_singles = max(max1, max2)

    fig, axes = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)
    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1/3, box_alpha=0, color='white')
    im0 = axes[0, 0].imshow(speckle_scan.single2s, vmin=0, vmax=max_singles)
    axes[0, 0].axis('off')
    axes[0, 0].add_artist(scalebar)
    # fig.colorbar(im0, ax=axes[0, 0])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im1 = axes[1, 0].imshow(speckle_scan.real_coins, vmin=min_coin, vmax=max_coin)
    axes[1, 0].axis('off')
    axes[1, 0].add_artist(scalebar)
    # fig.colorbar(im1, ax=axes[1, 0])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im2 = axes[0, 1].imshow(optimized_scan.single2s, vmin=0, vmax=max_singles)
    axes[0, 1].axis('off')
    axes[0, 1].add_artist(scalebar)
    fig.colorbar(im2, ax=axes[0, 1])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im3 = axes[1, 1].imshow(optimized_scan.real_coins, vmin=min_coin, vmax=max_coin)
    axes[1, 1].axis('off')
    axes[1, 1].add_artist(scalebar)
    fig.colorbar(im3, ax=axes[1, 1])


    fig.show()


def show_speckles(path1, path2, path3):
    scan1 = ScanResult(path1)
    scan2 = ScanResult(path2)
    scan3 = ScanResult(path3)

    max_singles = max(scan1.single2s.max(), scan2.single2s.max(), scan3.single2s.max())
    max_coin = max(scan1.real_coins.max(), scan2.real_coins.max(), scan3.real_coins.max())
    min_coin = max(scan1.real_coins.min(), scan2.real_coins.min(), scan3.real_coins.min())

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1/3, box_alpha=0, color='white')
    im0 = axes[0, 0].imshow(scan1.single2s, vmin=0, vmax=max_singles)
    axes[0, 0].axis('off')
    axes[0, 0].add_artist(scalebar)
    # fig.colorbar(im0, ax=axes[0, 0])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im1 = axes[1, 0].imshow(scan1.real_coins, vmin=min_coin, vmax=max_coin)
    axes[1, 0].axis('off')
    axes[1, 0].add_artist(scalebar)
    # fig.colorbar(im1, ax=axes[1, 0])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im2 = axes[0, 1].imshow(scan2.single2s, vmin=0, vmax=max_singles)
    axes[0, 1].axis('off')
    axes[0, 1].add_artist(scalebar)
    # fig.colorbar(im2, ax=axes[0, 1])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im3 = axes[1, 1].imshow(scan2.real_coins, vmin=min_coin, vmax=max_coin)
    axes[1, 1].axis('off')
    axes[1, 1].add_artist(scalebar)
    # fig.colorbar(im3, ax=axes[1, 1])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im2 = axes[0, 2].imshow(scan3.single2s, vmin=0, vmax=max_singles)
    axes[0, 2].axis('off')
    axes[0, 2].add_artist(scalebar)
    fig.colorbar(im2, ax=axes[0, 2])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0, color='white')
    im3 = axes[1, 2].imshow(scan3.real_coins, vmin=min_coin, vmax=max_coin)
    axes[1, 2].axis('off')
    axes[1, 2].add_artist(scalebar)
    fig.colorbar(im3, ax=axes[1, 2])

    fig.show()



path_not_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\2022_12_27_15_52_37_for_optimization_integration_8s_all'
path_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\2022_12_19_02_50_01_optimization_integration_5s_all_same'
# show_optimization(path_heralded)
# show_optimization(path_not_heralded)

path1_not_heralded = r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\Many speckles\2022_12_28_15_50_55_6_speckles.scan"
path2_not_heralded = r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\Many speckles\2022_12_28_15_50_55_4_speckles.scan"
path3_not_heralded = r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\Many speckles\2022_12_28_15_50_55_5_speckles.scan"
show_speckles(path1_not_heralded, path2_not_heralded, path3_not_heralded)
