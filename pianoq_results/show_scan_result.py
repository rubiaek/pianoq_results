import matplotlib as mpl
import matplotlib.pyplot as plt
from scan_result import ScanResult

import os
import sys

# mpl.use('QT5Agg')

path = sys.argv[1]
name = os.path.basename(path)

r = ScanResult()
r.loadfrom(path)
print(f'integration_time: {r.integration_time}')
r.show(show_singles=True, remove_accidentals=True,  title=name)
r.show(show_singles=False, remove_accidentals=False,  title=name)
plt.show()