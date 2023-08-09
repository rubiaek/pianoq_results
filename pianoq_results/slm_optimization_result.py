import matplotlib.pyplot as plt
import numpy as np

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

        # axes[1, 0].imshow(self.phase_masks[-1], cmap='gray')
        # fig.show()
        # fig.canvas.flush_events()

    def show_optimization_review(self):
        fig, ax = plt.subplots()
        ax.plot(self.costs)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Power (Arb. units)")
        return fig, ax

    @property
    def best_phase_mask(self):
        ind = np.argmax(self.costs)
        return self.phase_masks[ind]

