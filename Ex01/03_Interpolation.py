### Interpolation

import numpy as np
from matplotlib import pylab as plt

# Ex 3.1
def l(q, i, x):
    """returns the ith Lagrangian polynomial associated to the grid q,
    evaluated at x"""
    return np.prod([(x - q[j])/(q[i] - q[j])
                    for j in range(len(q)) if j!=i])

# Ex 3.2
def L(q, g, x):
    """returns the Lagrangian interpolation of g associated to the grid q,
    evaluated at x"""
    return sum([g(q[i])*l(q,i,x) for i in range(len(q))])

# Ex 3.3 - 3.4
for N in range(1, 14, 4):
    grid = np.linspace(0, 1, N)
    plt.plot([L(list(grid), lambda t : 1/(1 + (10*t - 5)**2), x)
              for x in np.linspace(0, 1, 1000)])
    plt.plot([1/(1 + (10*x - 5)**2) for x in np.linspace(0, 1, 1000)])
    plt.title("Degree "+ str(N) +" lagrangian approximation vs Runge function")
    plt.show()

#Ex 3.5
import numpy.polynomial as P
    
def lP(q, i):
    """returns the ith Lagrangian polynomial associated to the grid q,
    implemented with numpy.polynomial"""
    lPoly = P.Polynomial([1])
    for j in range(len(q)):
        if j!=i:
            den = (q[i]-q[j]) # the denominator of lagrangian polynomials
            lPoly *= P.Polynomial([1/den, -q[i]/den])
    return lPoly

def LP(q, g):
    """returns the Lagrangian interpolation of g associated to the grid q,
    implemented with numpy.polynomial"""
    LPoly = P.Polynomial([0])
    for j in range(len(q)):
        LPoly += g(q[j]) * lP(q,j)
    return LPoly

evalPoints = [h * 2**(-6) for h in range(2**6)]
interPoints = [h * 2**(-2) for h in range(2**2)]
DlP = lP(interPoints, 1).deriv(2)
# The 2nd lagrange polynomial with 4 interpolation points in [0,1] is
# 32 - 24x + 6x^2 - .5x^3; so its 2nd derivative should be 12 - 3x
plt.plot(evalPoints, [DlP(x) for x in evalPoints])
plt.title("2nd derivative of 32-24x+6x^2-.5x^3; indeed it's 12-3x")
plt.show()

# Ex 3.6
def Lambda(q, x):
    return sum([abs(l(q, i, x)) for i in range(len(q))])

for N in range(1, 14, 4):
    plt.plot([Lambda(np.linspace(0, 1, N), x)
              for x in np.linspace(0, 1, 1000)])
    plt.title("Degree " + str(N) + " Lambda function")
    plt.show()
# The oscillations on the edges get bigger as N increases.
# We can damp the oscillations adding more point on the edges;
# a quantitative estimate on where and how many points to add
# to obtain the "best" dampening is given by Chebyshev nodes.

# Ex 3.7
from scipy.optimize import minimize_scalar

def LambdaMaxList(q):
    """returns an approximation of the max of the Lambda function
    and the point in which it is attained, using minize_scalar from scipy
    in each interval of the grid q"""
    gridMax = [0, 0]
    
    for j in range(len(q)-1):
        # We will use minimize_scalar looping on all intervals [q[j], q[j+1]]
        start = q[j]
        end = q[j+1]
        midpoint = (q[j]+q[j+1])/2
        
        localMinInfo = minimize_scalar(
            lambda x : -Lambda(q,x),
            bracket = (start, midpoint, end))
        
        localMaxValue = abs(localMinInfo.get("fun"))       
        if localMaxValue > gridMax[0]:
            gridMax = [localMaxValue, localMinInfo.get("x")]
            # At each step we check, and eventually update, where Lambda
            # is maximum and the point in which the max is attained
            
    return gridMax

# Ex 3.8
def greedyStepMinimizeLambda(M, q):
    """given a starting grid q of [0,1], returns the M grid points
    obtained by adding points to q, at each addition requiring that
    a new point is placed where Lambda is maximum"""
    for j in range(M):
        q.append(LambdaMaxList(q)[1])
    return q

startGrid = [0, 0.5, 1]   # The starting grid of points
NPoints = 15              # Number of points which will be added to the grid
finalGrid = greedyStepMinimizeLambda(NPoints, startGrid)
plt.plot(finalGrid, [0 for i in range(len(finalGrid))] ,'ro')
plt.title("Greedy step minimization for "
          + str(len(finalGrid)) + " nodes")
plt.show()
# As the graph shows, indeed the nodes tend to concetrate on the edges
