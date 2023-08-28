from pianoq_results.klyshko_result import KlyshkoResult, show_memory

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
    res.show_optimization_process()
    res.show_best_phase()


def optimization():
    res = KlyshkoResult()
    res.loadfrom(PATH_OPTIMIZATION)
    res.print()
    res.show()
    res.show_optimization_process()
    res.show_best_phase()


def main():
    optimization()
    thick()
    memory()


if __name__ == '__main__':
    main()
