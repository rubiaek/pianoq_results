import sys
import numpy as np

path = sys.argv[1]
data = np.load(path)
s1 = data['single1s']
s2 = data['single2s']
s3 = data['single3s']
c1 = data['coin1s']
c2 = data['coin2s']

real_c1 = c1 - 2*s1*s2*1e-9
real_c2 = c2 - 2*s1*s3*1e-9

print(f'real c1: {real_c1.mean()}+-{c1.std() / np.sqrt(len(real_c1))}')
print(f'real c2: {real_c2.mean()}+-{c2.std() / np.sqrt(len(real_c2))}')
input()
