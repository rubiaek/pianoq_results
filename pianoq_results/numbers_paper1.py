import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pianoq_results.scan_result import ScanResult
from pianoq_results.piano_optimization_result import PianoPSOOptimizationResult


def optimization(heralded=False):
    path_not_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\2022_12_27_15_52_37_for_optimization_integration_8s_all'
    path_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\2022_12_19_02_50_01_optimization_integration_5s_all_same'

    # load data
    dir_path = path_heralded if heralded else path_not_heralded
    path = glob.glob(f'{dir_path}\\*speckles.scan')[0]
    speckle_scan = ScanResult(path)
    path = glob.glob(f'{dir_path}\\*optimized.scan')[0]
    optimized_scan = ScanResult(path)
    path = glob.glob(f'{dir_path}\\*.pqoptimizer')[0]
    optimization_res = PianoPSOOptimizationResult()
    optimization_res.loadfrom(path)

    # total optimization enhancement
    mean_cost_before = np.abs(optimization_res.all_costs[:optimization_res.n_for_average_cost].mean())
    best_cost_optimization = np.abs(optimization_res.all_costs.min())
    # print(f'Total Enhancement from pqoptimizer: {best_cost_optimization / mean_cost_before:.1f}')
    best_cost_from_scan = optimized_scan.real_coins.max()
    print(f'Total Enhancement from scan: {best_cost_from_scan / mean_cost_before:.1f}')

    # normalized optimization enhancement
    normalized_before = mean_cost_before / speckle_scan.real_coins.sum()
    normalized_after = best_cost_from_scan / optimized_scan.real_coins.sum()
    print(f'Normalized Enhancement from scan: {normalized_after / normalized_before:.1f}')

    # static single counts enhancement
    static_singles_before = speckle_scan.single1s.mean()
    static_singles_after = optimized_scan.single1s.mean()
    print(f'Static single counts enhancement: {static_singles_after / static_singles_before}')

    # Loss enhancement
    if not heralded:
        mask = np.index_exp[840:1200, 1750:1940]
    else:
        mask = np.index_exp[840:1200, 1500:1690]
    path = glob.glob(f'{dir_path}\\*singles_before.fits')[0]
    f = fits.open(path)[0]
    dark = f.data[:100, :100].mean()
    img_before = f.data[mask] - dark
    tot_power_before = img_before.sum()
    mimshow(img, f'before, heralded={heralded}')

    path = glob.glob(f'{dir_path}\\*_singles_after_optimization.fits')[0]
    f = fits.open(path)[0]
    dark = f.data[:100, :100].mean()
    img_after = f.data[mask] - dark
    tot_power_after = img_after.sum()
    mimshow(img, f'after, heralded={heralded}')

    path = glob.glob(f'{dir_path}\\*_singles_after_scan.fits')[0]
    f = fits.open(path)[0]
    dark = f.data[:100, :100].mean()
    img_after2 = f.data[mask] - dark
    tot_power_after2 = img_after2.sum()

    print(f'Loss enhancement: {tot_power_after / tot_power_before:.2f}')
    print(f'Loss enhancement: {tot_power_after2 / tot_power_before:.2f}')

    # polarization enhancement
    img_V_before = img_before[:180, :]
    img_H_before = img_before[180:, :]
    mimshow(img_V_before, 'V before')
    mimshow(img_H_before, 'H before')

    img_V_after = img_after[:180, :]
    img_H_after = img_after[180:, :]
    mimshow(img_V_after, 'V after')
    mimshow(img_H_after, 'H after')

    img_V_after2 = img_after2[:180, :]
    img_H_after2 = img_after2[180:, :]


## two spots ##
def two_spots():
    two_spots_herealded_path = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Two Spots\Heralded\2023_01_04_20_15_36_best_double_spot_2'
    two_spots_not_herealded_path = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Two Spots\Not Heralded\2023_01_08_11_51_22_double_spot_integration_2s_pretty_good'

def schmidt():
    pass


if __name__ == "__main__":
    print("##### one photon #####")
    optimization(True)
    print()

    print("##### two photons #####")
    optimization(False)


"""
TODO list: 
V enhancement optimization  
Schmidt  
enhancement two spots 

polarization enhancement 
V static single counts enhancement 
V normalized enhancement by total coincidences 
V loss enhancement 
"""
