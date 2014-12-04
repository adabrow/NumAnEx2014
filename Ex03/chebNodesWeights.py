import numpy as np
from scipy.interpolate import lagrange

def chebNodes(N):
    "Returns N Chebyshev nodes in (-1,1)."
    return np.array([np.cos(
        np.pi * (2*k-1)/(2*N)) for k in range(1,N+1)])

def l(j,q):
    "Returns the j-th lagrange basis polynomial on the nodes q."
    myf = len(q)*[0]
    myf[j] = 1
    return lagrange(q, myf)

def intL(j,q):
    """Returns the definite integral between -1, 1 of the the
    j-th lagrange basis polynomial on the nodes q."""
    return l(j,q).integ()(1) - l(j,q).integ()(-1)

def getWeights(q):
    "Returns weights associated with nodes q, using lagrange interp."
    return [intL(j,q) for j in range(len(q))]


# Documentation states that lagrange polynomials become numerically instable
# after 20 nodes; we try up to 30.
for n in range(30):
    for x in getWeights(chebNodes(n)):
        if x <= 0:
            print(n, x)
            raise Exception("Found negative weight.")
print("No negative weights found for n<30.\n")

# Actually for 37 nodes we get the first negative weights.
print("Negative weights for 37 Cheb nodes:")
cheb37 = getWeights(chebNodes(37))
for x in cheb37:
        if x <= 0:
            print(x)

# Is this due to numerical errors? To answer this question, notice that Cheb
# weights, as they were defined, should be symmetric with respect to 0.
# However for n=37:
print("\nThe following pairs should be equal:")
for j in range(18): 
    print(cheb37[j],cheb37[36-j])

