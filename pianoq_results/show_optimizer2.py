import os
import sys
import matplotlib.pyplot as plt
from slm_optimization_result import SLMOptimizationResult

path = sys.argv[1]
dirpath = os.path.dirname(path)
res = SLMOptimizationResult(dirpath)
res.print()
res.show_optimization_review()
plt.show()
