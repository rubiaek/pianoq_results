import glob
import matplotlib.pyplot as plt
from pianoq_results.scan_result import ScanResult
from matplotlib_scalebar.scalebar import ScaleBar


PATH1 = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\2022_12_27_15_52_37_for_optimization_integration_8s_all'
PATH2 = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\2022_12_19_02_50_01_optimization_integration_5s_all_same'


def show(dir_path):
    path = glob.glob(f'{dir_path}\\*speckles.scan')[0]
    speckle_scan = ScanResult(path)
    path = glob.glob(f'{dir_path}\\*optimized.scan')[0]
    optimized_scan = ScanResult(path)

    min1 = (speckle_scan.coincidences - speckle_scan.accidentals).min()
    min2 = (optimized_scan.coincidences - optimized_scan.accidentals).min()
    min_coin = min(min1, min2)
    print(min_coin)

    max_coin = (optimized_scan.coincidences - optimized_scan.accidentals).max()

    max1 = speckle_scan.single2s.max()
    max2 = optimized_scan.single2s.max()
    max_singles = max(max1, max2)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)
    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1/3, box_alpha=0)
    im0 = axes[0, 0].imshow(speckle_scan.single2s, vmin=0, vmax=max_singles)
    axes[0, 0].axis('off')
    axes[0, 0].add_artist(scalebar)
    fig.colorbar(im0, ax=axes[0, 0])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0)
    im1 = axes[1, 0].imshow(speckle_scan.coincidences - speckle_scan.accidentals, vmin=min_coin, vmax=max_coin)
    axes[1, 0].axis('off')
    axes[1, 0].add_artist(scalebar)
    fig.colorbar(im1, ax=axes[1, 0])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0)
    im2 = axes[0, 1].imshow(optimized_scan.single2s, vmin=0, vmax=max_singles)
    axes[0, 1].axis('off')
    axes[0, 1].add_artist(scalebar)
    fig.colorbar(im2, ax=axes[0, 1])

    scalebar = ScaleBar(25, units='um', location='upper right', length_fraction=1 / 3, box_alpha=0)
    im3 = axes[1, 1].imshow(optimized_scan.coincidences - optimized_scan.accidentals, vmin=min_coin, vmax=max_coin)
    axes[1, 1].axis('off')
    axes[1, 1].add_artist(scalebar)
    fig.colorbar(im3, ax=axes[1, 1])


    fig.show()


# show(PATH1)
show(PATH2)
