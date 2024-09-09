import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult

import os
import sys
path = sys.argv[1]
name = os.path.basename(path)


r = PhaseFinderResult()
r.loadfrom(path)

modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48]) - 1
print(r.phases[modes_to_keep])

fig, ax = plt.subplots()
ax.plot(r.phases[modes_to_keep], label=name)
fig.legend()
plt.show()