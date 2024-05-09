import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pianoq_results.QPPickleResult import QPPickleResult

LOGS_DIR = 'C:\\temp'


def show_phase_mask(A, title='diffuser'):
    fig, ax = plt.subplots()
    im = ax.imshow(A, cmap='gray')
    fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    fig.show()


class SLMOptimizationResult(QPPickleResult):
    def __init__(self, path=None, phase_masks=None, costs=None, cost_witnesses=None,
                 all_phase_masks=None, all_costs=None, all_cost_witnesses=None, opt_method=''):
        super().__init__(path=path)

        self.phase_masks = phase_masks or []
        self.costs = costs or []
        self.cost_witnesses = cost_witnesses or []
        self.all_phase_masks = all_phase_masks or []
        self.all_costs = all_costs or []
        self.all_cost_witnesses = all_cost_witnesses or []

        self.opt_method = opt_method
        self.best_phi_method = None
        self.roi = None

        # axes[1, 0].imshow(self.phase_masks[-1], cmap='gray')
        # fig.show()
        # fig.canvas.flush_events()

    def show_optimization_review(self, full=False):
        fig, ax = plt.subplots()
        if full:
            ax.plot(self.all_costs)
        else:
            ax.plot(self.costs)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Power (Arb. units)")
        fig.show()
        return fig, ax

    def show_before_after(self):
        if self.cost_witnesses:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
            fig.suptitle(f'Enhancement: {self.costs[-1] / self.costs[0]}')
            imm = axes[0].imshow(self.cost_witnesses[0])
            axes[0].set_title('Before')
            fig.colorbar(imm, ax=axes[0])
            imm2 = axes[1].imshow(self.cost_witnesses[-1])
            axes[1].set_title('After')
            fig.colorbar(imm2, ax=axes[1])
            if self.roi:
                if isinstance(self.roi, (list, tuple)) and isinstance(self.roi[0], (list, tuple)):
                    for roi in self.roi:
                        pass
                else:
                    rows = self.roi[0]
                    cols = self.roi[1]
                    axes[0].add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start,
                                                rows.stop - rows.start, facecolor='none', ec='r', lw=0.8))
                    axes[1].add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start,
                                                rows.stop - rows.start, facecolor='none', ec='r', lw=0.8))

                    N_rows = rows.stop - rows.start
                    N_cols = cols.stop - cols.start
                    A = 30
                    axes[0].set_ylim([rows.start - A * N_rows, rows.stop + A * N_rows])
                    axes[0].set_xlim([cols.start - A * N_rows, cols.stop + A * N_rows])

                    axes[1].set_ylim([rows.start - A * N_rows, rows.stop + A * N_rows])
                    axes[1].set_xlim([cols.start - A * N_cols, cols.stop + A * N_cols])

                pass # TODO: paint squares around cots region
            fig.show()
        else:
            print('no pictures')

    @property
    def best_phase_mask(self):
        ind = np.argmax(self.costs)
        return self.phase_masks[ind]

    @property
    def enhancement(self):
        return max(self.costs) / self.costs[0]

    def print(self):
        print(f'enhancement: {self.enhancement}')



# res = SLMOptimizationResult()
# res.loadfrom(r"G:\My Drive\Projects\ScalingPropertiesQWFS\Results\KlyshkoSetup\Try1\Between2Diffusers\2024_05_09_10_20_22_scaling_between_cell_size=40.optimizer2")
# res.show_before_after()
# plt.show()
