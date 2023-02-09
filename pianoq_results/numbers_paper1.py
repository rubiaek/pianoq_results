import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pianoq.misc.mplt import mimshow
from pianoq_results.scan_result import ScanResult
from pianoq_results.piano_optimization_result import PianoPSOOptimizationResult

def mimshow(*args, **kwargs):
    pass

def optimization(heralded=False):
    path_not_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\2022_12_27_15_52_37_for_optimization_integration_8s_all'
    path_heralded = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\2022_12_19_02_50_01_optimization_integration_5s_all_same'

    # load data
    if not heralded:
        dir_path = path_not_heralded
    else:
        dir_path = path_heralded

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
    print(f'Static single counts enhancement: {static_singles_after / static_singles_before:.1f}')

    # single counts enhancement
    index = np.unravel_index(optimized_scan.real_coins.argmax(), optimized_scan.real_coins.shape)
    singles_before = speckle_scan.single2s[index]
    singles_after = optimized_scan.single2s[index]
    print(f'Single counts scanning detctor enhancement: {singles_after / singles_before:.1f}')

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
    mimshow(img_before, f'before, heralded={heralded}')

    path = glob.glob(f'{dir_path}\\*_singles_after_optimization.fits')[0]
    f = fits.open(path)[0]
    dark = f.data[:100, :100].mean()
    img_after = f.data[mask] - dark
    tot_power_after = img_after.sum()
    mimshow(img_after, f'after, heralded={heralded}')

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
    V_H_ratio_before = img_V_before.sum() / img_H_before.sum()

    img_V_after = img_after[:180, :]
    img_H_after = img_after[180:, :]
    mimshow(img_V_after, 'V after')
    mimshow(img_H_after, 'H after')
    V_H_ratio_after = img_V_after.sum() / img_H_after.sum()

    img_V_after2 = img_after2[:180, :]
    img_H_after2 = img_after2[180:, :]
    V_H_ratio_after2 = img_V_after2.sum() / img_H_after2.sum()

    print(f'V/H ratio before: {V_H_ratio_before:.1f}')
    print(f'V/H ratio after: {V_H_ratio_after:.1f}')
    # print(f'V/H ratio after2: {V_H_ratio_after2}')


## two spots ##
def two_spots(heralded):
    two_spots_herealded_path = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Two Spots\Heralded\2023_01_04_20_15_36_best_double_spot_2'
    two_spots_not_herealded_path = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Two Spots\Not Heralded\2023_01_08_11_51_22_double_spot_integration_2s_pretty_good'

    # load data
    if not heralded:
        dir_path = two_spots_not_herealded_path
    else:
        dir_path = two_spots_herealded_path

    path = glob.glob(f'{dir_path}\\*optimized.scan')[0]
    scan = ScanResult(path)
    path = glob.glob(f'{dir_path}\\*config.json')[0]
    jjson = json.loads(open(path).read())
    path = glob.glob(f'{dir_path}\\*.randz')[0]
    randz_data = np.load(path)

    # Before
    s1 = randz_data['single1s']
    s2 = randz_data['single2s']
    s3 = randz_data['single3s']
    c1 = randz_data['coin1s']
    c2 = randz_data['coin2s']

    real_c1 = c1 - 2 * s1 * s2 * 1e-9
    real_c2 = c2 - 2 * s1 * s3 * 1e-9
    mean_before1 = real_c1.mean()  # lower spot
    mean_before2 = real_c2.mean()  # upper spot
    print(f'mean c1 before: {mean_before1:.1f}+-{c1.std() / np.sqrt(len(real_c1)):.1f}')
    print(f'mean c2 before: {mean_before2:.1f}+-{c2.std() / np.sqrt(len(real_c2)):.1f}')

    col, row = jjson['optimized_xy']
    col = np.where(scan.X == col)[0][0]
    row = np.where(scan.Y[::-1] == row)[0][0]

    # lower spot
    col2 = col + 2
    row2 = row + 7

    after1_opt1 = scan.real_coins2[row2, col2]  # lower spot
    after2_opt1 = scan.real_coins2[row, col]
    print(f'after1_opt1: {after1_opt1:.1f}')
    print(f'after2_opt1: {after2_opt1:.1f}')

    after1_opt2 = scan.real_coins2[row2-2:row2+3, col2-2:col2+3].max()  # lower spot
    after2_opt2 = scan.real_coins2[row-2:row+3, col-2:col+3].max()
    print(f'after1_opt2: {after1_opt2:.1f}')
    print(f'after2_opt2: {after2_opt2:.1f}')

    print('Option 1 - in place of X')
    print(f'enhancement lower spot: {after1_opt1 / mean_before1:.1f}')
    print(f'enhancement upper spot: {after2_opt1 / mean_before2:.1f}')
    print('Option 2 - maximum close to X')
    print(f'enhancement lower spot: {after1_opt2 / mean_before1:.1f}')
    print(f'enhancement upper spot: {after2_opt2 / mean_before2:.1f}')


def schmidt():
    pass


if __name__ == "__main__":
    print("######################")
    print("##### one photon #####")
    print("######################")
    optimization(True)
    print('\n## double spot ##')
    two_spots(True)

    print()
    print("######################")
    print("##### two photons ####")
    print("######################")
    optimization(False)
    print('\n## double spot ##')
    two_spots(False)


"""
TODO list: 
- all enhancements with uncertainty from shot noise   
- Schmidt number 
"""
