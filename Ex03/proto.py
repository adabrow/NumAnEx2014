import numpy as np
from numpy.polynomial.chebyshev import chebgauss
from matplotlib import pylab as plt

class Nodes:
    def __init__(self, nodes={}, a=0, b=1, **kwargs):
        if isinstance(nodes, dict):
            self.nodes = nodes.copy()
        elif isinstance(nodes, Nodes):
            self.nodes = {x:None for x in nodes.nodes.keys()}
        else:
            self.nodes = {x:None for x in nodes}

        self.a=a
        self.b=b

        if "equi" in kwargs.keys():
            self.nodes = {x : None for x in Nodes.eqNodes(
                kwargs["equi"]["nNodes"], self.a, self.b,
                kwargs["equi"]["extrIncl"])}

        if "cheb" in kwargs.keys():
            self.nodes = {x : None for x in Nodes.chebNodes(
                kwargs["cheb"]["nNodes"], self.a, self.b,
                kwargs["cheb"]["extrIncl"])}
            

    def eqNodes(N, a=0, b=1, extrIncl=True):
        """returns N equispaced nodes in [a,b].
    
        If $extrIncl is True a,b are included, otherwise not."""

        if extrIncl:
            return np.linspace(a,b,N)
        else:
            out = np.linspace(a,b,N+2)
            return list(np.array_split(out,[1,len(out)-1])[1])
            # subdivides an array in subarrays containing respectively
            # a,center,b and returns center (i.e. a linspace without extrema).


    def chebNodes(N, a=0, b=1, extrIncl=False):
        """returns N Chebyshev nodes in [a,b].
    
        The interval extrema are extracted from the tuple $extrema.
        If $extrIncl is True, a and b are included, otherwise not."""
    
        cheb = chebgauss(N)[0]*(b-a)/2 + (a+b)/2
    
        if not(extrIncl):
            return chebgauss(N)[0]*(b-a)/2 + (a+b)/2
        else:
            out = list(chebgauss(N-2)[0]*(b-a)/2 + (a+b)/2)
            out.append(a)
            out.reverse()
            out.append(b)
            return out
            
            
class WNodes(Nodes):

    def __init__(self, a=0, b=1, **kwargs):
        if "nodes" in kwargs.keys():
            _kwargs = kwargs.copy()
            del _kwargs["nodes"]
            
            if (not(isinstance(kwargs["nodes"],Nodes)) and
                "weights" in kwargs.keys()):
                
                _nodes = {kwargs["nodes"][i] : kwargs["weights"][i]
                               for i in range(len(kwargs["nodes"]))}
                super().__init__(_nodes, a, b, **_kwargs)
                
            else: super().__init__(kwargs["nodes"], a, b, **_kwargs)
            
        else: super().__init__({}, a, b, **kwargs)


    def equiSub(self, N):
        _nodes = [((x-self.a)/N, y) for x,y in sorted(self.nodes.items())]
        self.nodes = {}
        for j in range(N):
            for i in range(len(_nodes)):
                (self.nodes)[j/N + _nodes[i][0] + self.a] = _nodes[i][1]


    # shorthands for utilities

    def uniformWeights(self, val):
        for x in self.nodes.keys():
            self.nodes[x] = val
    
    def clear(self):
        self.nodes = {}

    def getList(self):
        return sorted(self.nodes)


from scipy.interpolate import lagrange 


class Quad(WNodes):

    def __init__(self, a=0, b=1, quadType=None, **kwargs):
        super().__init__(a,b,**kwargs)
        if quadType == "lagrange":
            _nodes = {}
            for x in self.nodes.keys():
                _nodes[x] = self.lagrangeWeight(x)
            self.nodes = _nodes
        

    def I(self, f):
        return sum(f(q)*w for q,w in self.nodes.items())


    def lagrangeBasis(self, x):
        "Returns the Lagrange interpolant with value 1 in $node"

        xGrid = self.getList()
        yGrid = [0]*len(self.nodes)
        i = xGrid.index(x)
        yGrid[i] = 1
        return lagrange(xGrid, yGrid)

    
    def lagrangeWeight(self,x):
        "Returns lagrange interpolant (with value 1 in $x) weight"
        return (self.lagrangeBasis(x).integ()(self.b)-
                self.lagrangeBasis(x).integ()(self.a))


# TESTING


NMax = 15

for test in [(lambda x : np.exp(-x), 1-1/np.exp(1)),
             (lambda x : 1/(1 + (10*x - 5)**2), np.arctan(5)/5)]:
    ya = []
    yb = []
    yc = []
                            
    for n in range(1,NMax):
        NewtonCotes = Quad(quadType="lagrange",
                           nodes=Nodes(equi={"nNodes":n, "extrIncl":True}))
        ya.append(NewtonCotes.I(test[0]))
    
        chebInterp = Quad(quadType="lagrange",
                           nodes=Nodes(cheb={"nNodes":n, "extrIncl":False}))
        yb.append(chebInterp.I(test[0]))

        iterTrapez = Quad(nodes=Nodes(equi={"nNodes":n+1, "extrIncl":True}))
        iterTrapez.uniformWeights(1/n)
        iterTrapez.nodes[0], iterTrapez.nodes[1] = 1/(2*n), 1/(2*n)
        yc.append(iterTrapez.I(test[0]))

    plt.semilogy(range(1,NMax), abs(np.array(ya)-test[1]), 'g')
    plt.semilogy(range(1,NMax), abs(np.array(yb)-test[1]), 'g')
    plt.semilogy(range(1,NMax), abs(np.array(yc)-test[1]), 'g')
    plt.show()
    

    
    
