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
print(f'mean singles1s: {r.single1s.mean()}')
print(f'first singles1s: {r.single1s[0, 0]}')
print(f'mean singles2s: {r.single2s.mean()}')
print(f'first singles2s: {r.single2s[0, 0]}')
r.show_singles()
r.show()
# if not r.is_double_spot:
#     r.show_good()
# else:
#     r.show_good_double()
# r.show(show_singles=True, remove_accidentals=True,  title=name)
# r.show(show_singles=False, remove_accidentals=False,  title=name)
plt.show()
