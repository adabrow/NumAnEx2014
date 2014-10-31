### Polynomials

import numpy as np
from matplotlib import pylab as plt

## Ex 2.1
grid = [0.995 + k*0.0001 for k in range(101)]
# the grid of points on which we calculate our function

plt.plot(grid, [(1-j)**6 for j in grid])
plt.plot(grid, [(j**6 - 6*j**5 + 15*j**4 - 20*j**3 + 15*j**2 - 6*j + 1)
            for j in grid])
plt.title("(1-x)^6 vs expanded expression near 1")
plt.show()
