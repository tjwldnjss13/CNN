import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7])
b = np.array([2, 2, 2, 2, 5, 5, 5])

for i in range(len(a)):
    if a[i] == b[i]:
        print(i)