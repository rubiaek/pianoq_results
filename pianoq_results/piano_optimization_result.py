import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

import traceback

from pianoq_results.misc import Player


class PianoPSOOptimizationResult(object):
    def __init__(self):
        # Results of experiment
        self.costs = np.array([])
        self.costs_std = []
        self.amplitudes = []
        self.images = []
        self.exposure_times = []
        self.timestamps = np.array([])

        # Dac params
        self.good_piezo_indexes = None
        self.max_piezo_voltage = None

        # Optimization params
        self.roi = None
        self.n_pop = None
        self.n_iterations = None
        self.stop_after_n_const_iters = None
        self.reduce_at_iterations = None

        self.normalized_images = []
        self.normaliztion_to_one = None
        self.random_average_cost = None
        self.n_for_average_cost = None
        self.cam_type = None

        self.all_costs = []
        self.all_costs_std = []
        self.all_amplitudes = []

    def _get_normalized_images(self):
        norm_ims = []
        for i, im in enumerate(self.images):
            norm_im = im / self.exposure_times[i]
            norm_im = norm_im / self.normaliztion_to_one
            norm_ims.append(norm_im)
        return norm_ims

    def show_image(self, im, title=None):
        fig, ax = plt.subplots()
        im = ax.imshow(im)
        fig.colorbar(im, ax=ax)
        if title:
            ax.set_title(title)
        fig.show()
        return fig, ax

    def show_result(self):
        fig, axes = plt.subplots(2, 1, figsize=(5, 5.8), constrained_layout=True)
        im0 = axes[0].imshow(self.normalized_images[0])
        im1 = axes[1].imshow(self.normalized_images[-1])
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        axes[0].set_title('Before')
        axes[1].set_title('After')

        fig.show()

    def plot_costs(self):
        fig, ax = plt.subplots()
        ax.errorbar(self.timestamps / 60, -self.costs, self.costs_std, fmt='o--', label='measurements', markersize=4)
        random_average_std = self.all_costs[:self.n_for_average_cost].std()
        ax.axhline(-self.random_average_cost, label='random average cost', color='g', linestyle='--')
        ax.axhspan(-self.random_average_cost - random_average_std, -self.random_average_cost + random_average_std,
                   alpha=0.4, color='g')
        ax.set_xlabel('time (min)')
        ax.set_ylabel('coincidence counts (1/s)')
        ax.legend()
        fig.show()

    def plot_amplitudes(self, amps):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim(0, 1)
        plt.xlabel("piezo_num")
        plt.ylabel("amplitude")

        ax.bar(range(len(self.good_piezo_indexes)), amps)  # , color = palette)
        fig.show()

    def animate_amplitudes(self):
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim(0, 1)
        palette = ['blue', 'red', 'green',
                   'darkorange', 'maroon', 'black']

        def animation_function(i):
            ax.clear()
            ax.set_ylim(0, 1)
            ax.set_title(f"Iteration {i}, Cost {self.costs[i]}")
            ax.set_xlabel("piezo_num")
            ax.set_ylabel("amplitude")

            ax.bar(range(len(self.good_piezo_indexes)), self.amplitudes[i])  # , color = palette)

        animation = Player(fig, animation_function, interval = 500, frames=len(self.amplitudes))
        plt.show()

    @property
    def enhancement(self):
        if self.random_average_cost:
            return np.min(self.costs) / self.random_average_cost
        else:
            return -1

    @property
    def power_in_L_Pol_before_after(self):
        im1 = self.normalized_images[0]
        middle = im1.shape[1] // 2
        percentage1 = im1[:, :middle].sum() / im1.sum()
        # print(f'initial percentage: {percentage1}')

        i = np.argmin(self.costs)
        im2 = self.normalized_images[i]
        percentage2 = im2[:, :middle].sum() / im2.sum()
        # print(f'final percentage: {percentage2}')

        return percentage1, percentage2

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     costs=self.costs,
                     costs_std=self.costs_std,
                     amplitudes=self.amplitudes,
                     images=self.images,
                     exposure_times=self.exposure_times,
                     timestamps=self.timestamps,
                     random_average_cost=self.random_average_cost,
                     n_for_average_cost=self.n_for_average_cost,
                     good_piezo_indexes=self.good_piezo_indexes,
                     max_piezo_voltage=self.max_piezo_voltage,
                     roi=self.roi,
                     n_pop=self.n_pop,
                     n_iterations=self.n_iterations,
                     stop_after_n_const_iters=self.stop_after_n_const_iters,
                     reduce_at_iterations=self.reduce_at_iterations,
                     cam_type=self.cam_type,
                     all_costs=self.all_costs,
                     all_costs_std=self.all_costs_std,
                     all_amplitudes=self.all_amplitudes
            )
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.costs = data['costs']
        self.costs_std = data.get('costs_std', [])
        self.amplitudes = data['amplitudes']
        self.images = data['images']
        self.exposure_times = data['exposure_times']
        self.timestamps = data['timestamps']
        self.random_average_cost = data.get('random_average_cost', None)
        self.n_for_average_cost = data.get('n_for_average_cost', 20)  # before the parameter was introduced it was always 20
        if self.n_for_average_cost != 20:
            self.n_for_average_cost = self.n_for_average_cost.item()
        if self.random_average_cost:
            self.random_average_cost = self.random_average_cost.item()

        # In classical optimization. In quantum - we don't have images, only costs
        if self.images[0] is not None:
            self.normaliztion_to_one = self.images[0].max() / self.exposure_times[0]
            self.normalized_images = self._get_normalized_images()

        self.good_piezo_indexes = data.get('good_piezo_indexes', None)
        self.max_piezo_voltage = data.get('max_piezo_voltage', None)

        self.roi = data.get('roi', None)
        self.n_pop = data.get('n_pop', None)
        self.n_iterations = data.get('n_iterations', None)
        self.stop_after_n_const_iters = data.get('stop_after_n_const_iters', None)
        self.reduce_at_iterations = data.get('reduce_at_iterations', None)
        self.cam_type = data.get('cam_type', None)

        self.all_costs = data.get('all_costs', [])
        self.all_costs_std = data.get('all_costs_std', [])
        self.all_amplitudes = data.get('all_amplitudes', [])
