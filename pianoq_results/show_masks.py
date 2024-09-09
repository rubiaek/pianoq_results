import matplotlib.pyplot as plt
from pianoq.simulations.mplc_sim.mplc_sim_result import MPLCMasks
import sys
import os

path = sys.argv[1]
name = os.path.basename(path)

msks = MPLCMasks()
msks.loadfrom(path)
msks.show()
plt.show()