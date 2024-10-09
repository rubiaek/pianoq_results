import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
import typing
import pyperclip
import traceback
import glob
from pianoq_results.misc import my_mesh, figshow
import datetime
import os


LOGS_DIR = "C:\\temp"


class ScanResult(object):
    def __init__(self, path=None, coincidences=None, coincidences2=None, single1s=None, single2s=None, single3s=None, X=None, Y=None,
                 is_double_spot=False, integration_time=None, coin_window=None, is_timetagger=None):
        self.path = path
        self.coincidences = coincidences
        self.coincidences2 = coincidences2
        self.single1s = single1s
        self.single2s = single2s
        self.single3s = single3s
        self.is_double_spot = is_double_spot
        self.accidentals = None
        self.X = X
        self.Y = Y
        self.integration_time = integration_time
        self.coin_window = coin_window
        self.is_timetagger = is_timetagger

        if self.path is not None:
            self.loadfrom(self.path)

    def show(self, show_singles=False, title='', remove_accidentals=True, ax=None) -> typing.Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        self.reload()
        if remove_accidentals:
            coin = self.coincidences - self.accidentals
        else:
            coin = self.coincidences

        if ax is None:
            fig, ax = plt.subplots()
        if remove_accidentals:
            rem_acc = 'no accidentals'
        else:
            rem_acc = 'with accidentals'
        ax.set_title(f'Coincidences {title} - {rem_acc}')
        ax.text(16.5, 16.2, f'max acc: {self.accidentals.max() :.0f} \n'
                      f'max std: {np.sqrt(self.accidentals.max() * self.integration_time) / self.integration_time :.0f}')
        my_mesh(self.X, self.Y, coin, ax)
        ax.invert_xaxis()
        ax.figure.show()

        if show_singles:
            fig, axes = plt.subplots(1, 2)
            my_mesh(self.X, self.Y, self.single1s, axes[0])
            my_mesh(self.X, self.Y, self.single2s, axes[1])
            axes[0].invert_xaxis()
            axes[1].invert_xaxis()
            axes[0].set_title(f'Single counts 1 {title}')
            axes[1].set_title(f'Single counts 2 {title}')
            fig.show()

        return ax.figure, ax

    @property
    def real_coins(self):
        return self.coincidences - self.accidentals

    @property
    def real_coins_std(self):
        return np.nan_to_num(np.sqrt(self.real_coins)) / np.sqrt(self.integration_time)

    @property
    def real_coins2(self):
        return self.coincidences2 - self.accidentals2

    def show_good(self, title=''):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].set_title(f'Single counts 2 {title}')
        my_mesh(self.X, self.Y, self.single2s, axes[0])

        axes[1].set_title(f'Coincidences {title} - no accidentals')
        my_mesh(self.X, self.Y, self.real_coins, axes[1])

        axes[0].invert_xaxis()
        axes[1].invert_xaxis()
        fig.show()

    def show_good_double(self, title=''):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
        axes[0, 0].set_title(f'Single counts 2 {title}')
        my_mesh(self.X, self.Y, self.single2s, axes[0, 0])

        axes[0, 1].set_title(f'Coincidences {title} - no accidentals')
        my_mesh(self.X, self.Y, self.real_coins, axes[0, 1])

        axes[1, 0].set_title(f'Single counts 3 {title}')
        my_mesh(self.X, self.Y, self.single3s, axes[1, 0])

        axes[1, 1].set_title(f'Coincidences2 {title} - no accidentals')
        my_mesh(self.X, self.Y, self.real_coins2, axes[1, 1])

        # axes[0, 0].invert_xaxis()
        # axes[0, 1].invert_xaxis()
        # axes[1, 0].invert_xaxis()
        # axes[1, 1].invert_xaxis()
        fig.show()

    @property
    def displacement_between_coins(self):
        correlation = correlate2d(self.real_coins2, self.real_coins)
        i, j = np.unravel_index(np.argmax(correlation), correlation.shape)
        displacement_pixels = i - self.real_coins.shape[0], j - self.real_coins.shape[1]
        dx = self.X[1] - self.X[0]
        dy = self.Y[1] - self.Y[0]
        displacement_mm = displacement_pixels[0] * dy, displacement_pixels[1] * dx
        return displacement_mm

    @property
    def extent(self):
        dx = (self.X[1] - self.X[0]) / 2
        dy = (self.Y[1] - self.Y[0]) / 2
        extent = (self.X[0] - dx, self.X[-1] + dx, self.Y[0] - dy, self.Y[-1] + dy)
        return extent

    def show_both(self):
        self.show(show_singles=True, remove_accidentals=True)
        self.show(show_singles=False, remove_accidentals=False)

    def show_singles(self, title='', only=None):
        self.reload()
        if not only:
            fig, axes = plt.subplots(1, 2)
            my_mesh(self.X, self.Y, self.single1s, axes[0])
            my_mesh(self.X, self.Y, self.single2s, axes[1])
            axes[0].invert_xaxis()
            axes[1].invert_xaxis()
            axes[0].set_title(f'Single counts 1 {title}')
            axes[1].set_title(f'Single counts 2 {title}')
            figshow(fig)
            return fig
        else:
            fig, ax = plt.subplots()
            if only == 1:
                my_mesh(self.X, self.Y, self.single1s, ax)
            elif only == 2:
                my_mesh(self.X, self.Y, self.single2s, ax)
            ax.invert_xaxis()
            ax.set_title(f'Single counts {only} {title}')
            figshow(fig)
            return fig

    def get_xys(self, single_num=1, num_spots=5, timeout=0, saveto_dir=None, saveto_path=None):
        if single_num == 1:
            s = self.single1s
            name = 'idl'
            txt = 'middle to up (lower y)'
        elif single_num == 2:
            s = self.single2s
            name = 'sig'
            txt = 'middle to down (higher y)'
        fig, ax = plt.subplots()
        my_mesh(self.X, self.Y, s, ax)
        ax.set_title(f'Single counts {single_num}, {name}, {txt}')
        fig.show()
        locs = fig.ginput(n=num_spots, timeout=timeout)

        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if saveto_path is None:
            saveto_dir = saveto_dir or "C:\\temp"
            saveto_path = os.path.join(saveto_dir, f'{timestamp}_s{single_num}_{name}.locs')

        f = open(saveto_path, 'wb')
        np.savez(f, locs=locs)
        f.close()
        print(f"Saved to {saveto_path}")

        return locs

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     coincidences=self.coincidences,
                     coincidences2=self.coincidences2,
                     single1s=self.single1s,
                     single2s=self.single2s,
                     single3s=self.single3s,
                     is_double_spot=self.is_double_spot,
                     X=self.X,
                     Y=self.Y,
                     integration_time=self.integration_time,
                     coin_window=self.coin_window,
                     is_timetagger=self.is_timetagger)
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path=None, BAK=True):
        if path is None:
            BAK = '' if not BAK else '.BAK'
            paths = glob.glob(f"{LOGS_DIR}\\*.scan")
            for i, path in enumerate(paths):
                print(f'{i}: {path}')
            choice = int(input('which one?'))
            path = paths[choice]
        elif path == 0:
            path = pyperclip.paste()

        path = path.strip('"')
        path = path.strip("'")
        self.path = path

        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.coincidences = data['coincidences']
        self.coincidences2 = data.get('coincidences2', 0)
        self.single1s = data['single1s']
        self.single2s = data['single2s']
        self.single3s = data.get('single3s', 0)
        self.is_double_spot = data.get('is_double_spot', False)
        if type(self.is_double_spot) is np.ndarray:
            self.is_double_spot = self.is_double_spot.item()

        self.X = data['X']
        self.Y = data['Y']
        self.integration_time = data.get('integration_time', None)
        self.coin_window = data.get('coin_window', 4e-9)
        self.is_timetagger = data.get('is_timetagger', False)

        self.accidentals = self.single1s * self.single2s * 2 * self.coin_window
        self.accidentals2 = self.single1s * self.single3s * 2 * self.coin_window

        f.close()

    def reload(self):
        self.loadfrom(self.path)
