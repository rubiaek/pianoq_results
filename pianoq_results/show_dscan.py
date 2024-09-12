import matplotlib.pyplot as plt
from pianoq.lab.mplc.discrete_scan_result import DiscreetScanResult
import sys
import os

path = sys.argv[1]
name = os.path.basename(path)

r = DiscreetScanResult()
r.loadfrom(path)
r.show()
r.show_singles()

plt.show()
