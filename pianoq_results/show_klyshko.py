import os
import sys
import matplotlib.pyplot as plt
from klyshko_result import KlyshkoResult

path = sys.argv[1]
dirpath = os.path.dirname(path)
res = KlyshkoResult(dirpath)
res.print()
res.show()
res.show_best_phase()
res.show_optimization_process()
plt.show()
