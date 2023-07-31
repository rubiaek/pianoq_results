import numpy as np
import matplotlib.pyplot as plt
import traceback
import pyperclip


def show_phase_mask(A, title='diffuser'):
    fig, ax = plt.subplots()
    im = ax.imshow(A, cmap='gray')
    fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    fig.show()


class OptimizerResult(object):
    def __init__(self, diffuser_phase_grid=None, original_speckle_pattern=None, best_result=None,
                 slm_phase_grid=None, powers=None, mid_results=None, original_good=None, opt_method=''):
        self.original_good = original_good
        self.diffuser_phase_grid = diffuser_phase_grid
        self.original_speckle_pattern = original_speckle_pattern
        self.best_result = best_result
        self.slm_phase_grid = slm_phase_grid
        self.powers = powers
        self.mid_results = mid_results
        self.opt_method = opt_method

    @property
    def enhacement(self):
        # Choose pixels using original focus
        print("## Relevant only for simulation!! ##")
        max_power = self.original_good.max()
        print('max_power:', max_power)
        limit_power = max_power / 4
        print('limit_power:', limit_power)
        relevant_pixels = self.original_good > limit_power
        original_tot_power = np.sum(self.original_good[relevant_pixels])
        print("original_tot_power", original_tot_power)

        if False:
            fig, ax = plt.subplots()
            ax.set_title("Relevant pixels")
            ax.imshow(relevant_pixels)
            fig.show()

        # Sum powers after fixing
        relevant_powers = self.best_result[relevant_pixels]
        # print("relevant_powers", relevant_powers)
        power_after_fix = np.sum(relevant_powers)
        print("power_after_fix", power_after_fix)

        # estimate random speckle intensity
        relevant_powers = self.original_speckle_pattern[relevant_pixels]
        # print("relevant_powers", relevant_powers)
        power_speckles = np.sum(relevant_powers)
        print("power_before_fix", power_speckles)

        enhancement = power_after_fix / power_speckles
        print('enhancement', enhancement)

        efficiency1 = power_after_fix / original_tot_power
        efficiency2 = (power_after_fix - power_speckles) / (original_tot_power - power_speckles)
        print("efficiency1", efficiency1)
        print("efficiency2", efficiency2)

        return enhancement

    def show_all(self):
        self.show_diffuser()
        self.show_orig()
        self.show()

    def show_diffuser(self):
        show_phase_mask(self.diffuser_phase_grid, 'phase grid on diffuser')

    def show_orig(self):
        fig, ax = plt.subplots()
        ax.imshow(self.original_good)
        ax.set_title('Before diffuser')
        fig.show()
        fig.canvas.flush_events()

    def show(self):
        fig, axes = plt.subplots(2, 2)
        fig.suptitle(f'Results Of {self.opt_method} Optimization')

        axes[0, 0].imshow(self.original_speckle_pattern)
        axes[0, 0].set_title('Original Speckle Pattern')

        axes[0, 1].imshow(self.best_result)
        axes[0, 1].set_title('Best Result')

        axes[1, 0].imshow(self.slm_phase_grid, cmap='gray')
        axes[1, 0].set_title('final phase grid on SLM')

        axes[1, 1].plot(self.powers)
        axes[1, 1].set_title("Power over Iterations")

        fig.show()
        fig.canvas.flush_events()

    def show_optimization_review(self):
        fig, ax = plt.subplots()
        ax.plot(self.powers)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Power (Arb. units)")
        return fig, ax

    @property
    def best_slm_phase_grid(self):
        best_mid_phase, best_mid_power = max(self.mid_results.values(), key=lambda x: x[1])
        if best_mid_power > self.powers[-1]:
            return best_mid_phase
        return self.slm_phase_grid

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     diffuser_phase_grid=self.diffuser_phase_grid,
                     original_speckle_pattern=self.original_speckle_pattern,
                     best_result=self.best_result,
                     slm_phase_grid=self.slm_phase_grid,
                     powers=self.powers,
                     mid_results=self.mid_results,
                     original_good=self.original_good,
                     opt_method=self.opt_method)
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path=None):
        if path is None:
            paths = glob.glob(f"{LOGS_DIR}\\*scan*.npz*")
            for i, path in enumerate(paths):
                print(f'{i}: {path}')
            choice = int(input('which one?'))
            path = paths[choice]
        elif path == 0:
            path = pyperclip.paste()

        path = path.strip('"')
        path = path.strip("'")

        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)

        self.diffuser_phase_grid = data['diffuser_phase_grid']
        self.original_speckle_pattern = data['original_speckle_pattern']
        self.best_result = data['best_result']
        self.slm_phase_grid = data['slm_phase_grid']
        self.powers = data['powers']
        try:
            self.mid_results = data['mid_results'].all()
            self.original_good = data['original_good']
            self.opt_method = data['opt_method']
        except Exception:
            print('old optimization')
        f.close()

