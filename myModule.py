import numpy as np
from numpy.polynomial.chebyshev import chebgauss

class Nodes:
    def getEq(N, extrIncl=True):
        """returns N equispaced nodes in [0,1].
        If extrIncl is True (default) 0,1 are included, otherwise not."""

        if extrIncl: return np.linspace(0,1,N)
        else:
            out = np.linspace(0,1,N+2)
            return np.array_split(out,[1,len(out)-1])[1]
            # subdivides an array in subarrays containing respectively
            # 0,center,1 and returns center (i.e. a linspace without extrema).


    def getCh(N, extrIncl=False):
        """returns N Chebyshev nodes in [0,1].

        If extrIncl is True 0,1 are included, otherwise not."""
        if not(extrIncl): return (chebgauss(N)[0]/2 +.5)[::-1]
        else:
            out = list(chebgauss(N-2)[0]/2 +.5)
            out.append(0)
            out.reverse()
            out.append(1)
            return np.array(out)
