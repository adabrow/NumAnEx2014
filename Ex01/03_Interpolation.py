### Interpolation

import numpy as np
from matplotlib import pylab as plt

# Ex 3.1
def l(q, i, x):
    """returns the ith Lagrangian polynomial 
    associated to the grid q, evaluated at x"""
    return np.prod([(x - q[j])/(q[i] - q[j])
                    for j in range(len(q)) if j!=i])

# Ex 3.2
def L(q, g, x):
    """returns the Lagrangian interpolation of g
    associated to the grid q, evaluated at x"""
    return sum([g(q[i])*l(q,i,x) for i in range(len(q))])

# Ex 3.3 - 3.4
for N in range(1, 14, 4):
    grid = np.linspace(0, 1, N)
    plt.plot([L(list(grid), lambda t : 1/(1 + (10*t - 5)**2), x)
              for x in np.linspace(0, 1, 1000)])
    plt.plot([1/(1 + (10*x - 5)**2) for x in np.linspace(0, 1, 1000)])
    plt.title("Degree " + str(N) + " lagrangian approximation vs Runge function")
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

def LambdaMinList(q):
    """returns an approximation of the max of the Lambda function
    and the point in which it is attained (using minize_scalar from scipy)"""
    return [abs(minimize_scalar(lambda x : -Lambda(q,x),
                               method='bounded', bounds=(0,1)).get("fun")),
                minimize_scalar(lambda x : -Lambda(q,x),
                               method='bounded', bounds=(0,1)).get("x")]

# Ex 3.8
def greedyStepMinimizeLambda(M, q):
    """given a starting grid q of [0,1],
    returns the M grid points obtained by adding points to q, and
    at each addition requiring that new point is placed where
    Lambda is maximum"""
    for j in range(M):
        q.append(LambdaMinList(q)[1])
    plt.plot(q, [0 for i in range(len(q))] ,'ro')
    plt.title("Smart step minimization for " + str(M) + " nodes")
    plt.show()
    return q

greedyStepMinimizeLambda(35, [0, 0.5, 1])
# As the graph shows, indeed the nodes tend to concetrate on the edges
