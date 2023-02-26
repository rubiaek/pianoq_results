import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import signal

from pianoq.misc.mplt import mimshow
from pianoq.lab.scripts.two_speckle_statistics import SpeckleStatisticsResult
from pianoq_results.scan_result import ScanResult
from pianoq_results.piano_optimization_result import PianoPSOOptimizationResult

# def mimshow(*args, **kwargs):
#     pass

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


def schmidt_before():
    # but this might not be correct
    Lambda = 404e-9
    n_ppktp = 1.82
    L = 4e-3
    k = 2*np.pi*n_ppktp / Lambda
    b = np.sqrt(L/(4*k))
    sigma = 1/(46*3.76e-6*200/300)
    K = 0.25*(b*sigma + 1/(b*sigma))
    print(f'Schmidt number from crystal and waist: {K}')


def schmidt_after():
    ss = SpeckleStatisticsResult()
    ss.loadfrom(r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\2023_01_01_20_06_20_speckle_statistics_filter=3nm_heralded_integration_5s.spst")
    cc, cdc = ss._get_contrast(ss.real_coin)
    s2c, s2dc = ss._get_contrast(ss.single2s)

    from uncertainties import ufloat
    cc = ufloat(cc, cdc)
    s2c = ufloat(s2c, s2dc)

    Nc = 1/cc**2
    Ns2 = 1/s2c**2

    print(f'Contrast for coincidence: {cc} ~ {Nc} speckle patterns')
    print(f'Contrast for single2s: {s2c} ~ {Ns2} speckle patterns')

    schmidt_number = Ns2 / Nc
    print(f"Schmidt number: {schmidt_number}")


def fiber_modes():
    from uncertainties import ufloat
    NA = ufloat(0.2, 0.015)
    r_core = ufloat(50e-6, 2.5e-6) / 2
    L = ufloat(808e-9, 3e-9)

    V = (2*np.pi/L)*NA*r_core
    N = (4/np.pi**2)*V**2
    print(f"N modes is: {N:.2f}")


def spectral_correlation_width():
    from pyspectra.readers.read_spc import read_spc
    from pianoq.misc.mplt import mplot
    import glob
    path = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Supplementary\Different spectral speckles with SLD and spectrometer\*.spc'
    files = glob.glob(path)
    Xs = []
    Ys = []
    for f in files:
        spc = read_spc(f)
        Xs.append(np.array(spc.index)[2510:2810])
        Ys.append(spc.values[2510:2810])

    dx = Xs[0][1] - Xs[0][0]

    mean_sig = np.mean(Ys, axis=0)
    auto_corrs = [np.correlate(V - V.mean(), V - V.mean(), mode='full') for V in Ys]
    auto_corrs = [atc / atc.max() for atc in auto_corrs]
    X = signal.correlation_lags(len(Ys[0]), len(Ys[0]), 'full')

    atc_of_mean = np.correlate(mean_sig - mean_sig.mean(), mean_sig - mean_sig.mean(), mode='full')
    atcc = atc_of_mean / atc_of_mean.max()
    final = np.mean(auto_corrs, axis=0) / atc_of_mean
    mplot(X, final)  # FWHM -> spectral correlation width of 2.8nm


def loss():
    dir_path = r"G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Supplementary\PressesLoss\Amps=%s.fits"

    a0 = fits.open(dir_path % 0)[0]
    a05 = fits.open(dir_path % 0.5)[0]
    a1 = fits.open(dir_path % 1)[0]

    sum0 = (a0.data - a0.data[:, 0].mean()).sum()
    sum05 = (a05.data - a05.data[:, 0].mean()).sum()
    sum1 = (a1.data - a1.data[:, 0].mean()).sum()

    sum0 = sum0 / a0.header['expoinus']
    sum05 = sum05 / a05.header['expoinus']
    sum1 = sum1 / a1.header['expoinus']

    print(f'sum0   : {sum0:.0f}')
    print(f'sum05  : {sum05:.0f}')
    print(f'sum1   : {sum1:.0f}')

    ################################
    dir_path = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Supplementary\SinglesWithAndWithoutPiano'
    path_in_piano = glob.glob(f'{dir_path}\\*inside_piano.fit')[0]
    path_out_piano = glob.glob(f'{dir_path}\\*outside_piano.fit')[0]

    f_out = fits.open(path_out_piano)[0]
    data_out = f_out.data[800:1300, 1550:1850]
    data_out = data_out - data_out[:, 0].mean()

    f_in = fits.open(path_in_piano)[0]
    data_in = f_in.data[800:1300, 1550:1850]
    data_in = data_in - data_in[:, 0].mean()

    sum_in = data_in.sum() / (f_in.header['expoinus']*1e-6)
    sum_out = data_out.sum() / (f_out.header['expoinus']*1e-6)

    print(f'sum_in : {sum_in:.0f}')
    print(f'sum_out: {sum_out:.0f}')


def main_article_numbers():
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


if __name__ == "__main__":
    loss()

"""
TODO list: 
- all enhancements with uncertainty from shot noise   
- Spectral correlation width  
"""
