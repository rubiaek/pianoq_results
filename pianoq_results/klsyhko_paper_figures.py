import matplotlib.pyplot as plt
from pianoq_results.klyshko_result import KlyshkoResult, show_memory
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage
from pianoq_results.misc import my_mesh

PATH_OPTIMIZATION = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_24_14_54_48_klyshko_two_diffusers_other_0.25_0.5_power_meter_continuous_hex_in_place'
PATH_THICK = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_28_08_10_25_klyshko_thick_diffuser_0.25_and_0.25_0.16_power_meter_continuous_hex'
PATH_THICK_MEMORY = r'G:\My Drive\Projects\Klyshko Optimization\Results\Good\2023_08_28_08_10_25_klyshko_thick_diffuser_0.25_and_0.25_0.16_power_meter_continuous_hex\memory_measurements'


def memory():
    show_memory(PATH_THICK_MEMORY, show_ds=(50, 75, 150, 250, 350), classic=True)
    show_memory(PATH_THICK_MEMORY, show_ds=(50, 75, 150, 250), classic=False)


def thick():
    res = KlyshkoResult()
    res.loadfrom(PATH_THICK)
    res.print()
    res.show()
    # res.show_optimization_process()
    # res.show_best_phase()


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
    diode = FITSImage(r"G:\My Drive\Projects\Klyshko Optimization\Results\Good\Speckles same\2023_08_21_10_19_52_diode_speckles.fits")
    imm = axes[0].imshow(diode.image)
    axes[0].set_xlim(left=170, right=420)
    axes[0].set_ylim(top=120, bottom=370)

    fig.colorbar(imm, ax=axes[0])
    SPDC = ScanResult(r"G:\My Drive\Projects\Klyshko Optimization\Results\Good\Speckles same\2023_08_21_11_27_52_scan_trying_Klyshko_1_diffuser_slm_optimized.scan")
    my_mesh(SPDC.X, SPDC.Y, SPDC.real_coins, axes[1])
    axes[1].invert_xaxis()
    fig.show()


def main():
    optimization()
    thick()
    memory()
    similar_speckles()


if __name__ == '__main__':
    main()
