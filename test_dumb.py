import numpy as np

x = np.array([1,2,3])
y = np.copy(x)
x[0]  = 10
print(y)