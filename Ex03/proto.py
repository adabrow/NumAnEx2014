import numpy as np
from numpy.polynomial.chebyshev import chebgauss
from matplotlib import pylab as plt
from scipy.interpolate import lagrange
from numpy.polynomial.legendre import leggauss

def posit(x):
        if x < 0: return 0
        if x >=0: return x

class _Nodes:
    # Not really intended for use; it is a blueprint for WNodes which contains
    # features independent from weights behavior.
    """Represents nodes as keys of a dictionary (in the subclass WNodes
    the values become weights associated to the node).
    __init__ handles arguments with the following behavior:
    $wnodes is expected to be a dictionary (in which case is copied into a 
    variable), or a list (becomes a dictionary with uniform values None),
    or another _Nodes object (its keys are copied);
    $a, $b are the extrema of the interval in which all nodes should be in;
    $equi is an optional argument used to generate equispaced nodes in [a,b],
    which is expected to be a dictionary containing a key nNodes with value
    the number of nodes, and an optional key extrIncl with value True or False
    depending if extrema should be included or not in the generated nodes.
    $cheb is an optional argument used to generate Chebyshev nodes in [a,b],
    which is expected to be a dictionary with the same assumptions of $equi.
    """
    
    def __init__(self, wnodes={}, a=0, b=1, **kwargs):
        """Inputs should be codified as follows:
        for $wnodes, a dictionary with keys as nodes and None as values;
        for $a and $b the extrema of the interval in which all nodes sit;
        a dictionary containing more options (see also _Nodes.__doc___)."""
        
        if isinstance(wnodes, dict):
            self.wnodes = wnodes.copy()
        elif isinstance(wnodes, _Nodes):
            self.wnodes = {x:None for x in wnodes.wnodes.keys()}
        else:
            self.wnodes = {x:None for x in wnodes}

        self.a=a
        self.b=b

        if "equi" in kwargs.keys():
            if "extrIncl" in kwargs["equi"].keys():
                extrIncl = kwargs["equi"]["extrIncl"]
            else: # defaults to True
                extrIncl = True 

            self.wnodes = {x : None for x in _Nodes.eqNodes(
                kwargs["equi"]["nNodes"], self.a, self.b, extrIncl)}

        if "cheb" in kwargs.keys():
            if "extrIncl" in kwargs["cheb"].keys():
                extrIncl = kwargs["cheb"]["extrIncl"]
            else: # defaults to False
                extrIncl = False 

            self.wnodes = {x : None for x in _Nodes.chebNodes(
                kwargs["cheb"]["nNodes"], self.a, self.b, extrIncl)}
            

    # Some handy "static" methods

    def eqNodes(N, a=0, b=1, extrIncl=True):
        """return N equispaced nodes in [a,b].
    
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
    
        If $extrIncl is True, a,b are included, otherwise not."""
    
        cheb = chebgauss(N)[0]*(b-a)/2 + (a+b)/2
    
        if not(extrIncl):
            return chebgauss(N)[0]*(b-a)/2 + (a+b)/2
        else:
            out = list(chebgauss(N-2)[0]*(b-a)/2 + (a+b)/2)
            out.append(a)
            out.reverse()
            out.append(b)
            return out


    # utilities

    def clear(self):
        self.wnodes = {}

    def getList(self):
        return sorted(self.wnodes)
            
            
class WNodes(_Nodes):
    """Represents nodes and weights resp. as keys and values in a single
    dictionary $wnodes.
    __init__ handles arguments as in _Nodes, with the addition of $weights
    optional argument, which, given a iterable collection of wnodes, tries
    to associate to them the weights specified in $weights."""

    def __init__(self, wnodes={}, a=0, b=1, **kwargs):
            
        if "weights" in kwargs.keys():
            if len(wnodes) != 0:                
                _wnodes = {wnodes[i] : kwargs["weights"][i]
                                   for i in range(len(wnodes))}
                self.wnodes = _wnodes
                super().__init__(self.wnodes, a, b, **kwargs)
                
            else: # if there are no explicit nodes tries to build them through
                  # super (hoping that optional arguments will generate them),
                  # then runs self.__init__ without options (except weights);
                  # if no new nodes are obtained, just runs super().__init__.
                  
                newNodes = (_Nodes(wnodes, a, b, **kwargs)).getList()
                
                if len(newNodes) != 0:
                    self.__init__(newNodes, a, b, weights = kwargs["weights"])
                else:
                    super().__init__({}, a, b, **kwargs)
            
        else:
            super().__init__(wnodes, a, b, **kwargs)


    def iterNW(self, N):
        "Rescales actual wnodes by a factor 1/N"
        _wnodes = {q/N : w/N for q,w in self.wnodes.items()}
        self.wnodes = {q + j*N : w for j in range(N) for q,w in _wnodes.items()}


    # shorthands for some utilities

    def equiSub(self, N):
        _wnodes = [((x-self.a)/N, y) for x,y in sorted(self.wnodes.items())]
        self.wnodes = {}
        for j in range(N):
            for i in range(len(_wnodes)):
                (self.wnodes)[j/N + _wnodes[i][0] + self.a] = _wnodes[i][1]

    def uniformWeights(self, val):
        for x in self.wnodes.keys():
            self.wnodes[x] = val
    


class Quad(WNodes):
    """Calculates weights and nodes following specific rules (e.g.
    Newton-Cotes) and approximates definite integrals of funcions.
    __init__ handles arguments as in _WNodes, with the following additions:
    getWeightsW argument with value the rule to get weights from nodes can
    be specified (currently supports only "lagrange");
    OpenNewtonCotes and ClosedNewtonCotes arguments with value the degree
    of the rule can be passed to calculate automatically the relative
    nodes and weights."""

    def __init__(self, wnodes={}, a=0, b=1, interpType=None,
                 iterations=0, **kwargs):

        if "OpenNewtonCotes" in kwargs:
            n = kwargs["OpenNewtonCotes"] # NewtonCotes' degree
            _kwargs = kwargs.copy()
            del _kwargs["OpenNewtonCotes"]
            _kwargs["equi"] = {"nNodes":n-1, "extrIncl":False}
            super().__init__(wnodes, a, b, **_kwargs)
            
        elif "ClosedNewtonCotes" in kwargs:
            n = kwargs["ClosedNewtonCotes"] # NewtonCotes' degree
            _kwargs = kwargs.copy()
            del _kwargs["ClosedNewtonCotes"]
            _kwargs["equi"] = {"nNodes":n+1, "extrIncl":True}
            super().__init__(wnodes, a, b, **_kwargs)

        else:
            super().__init__(wnodes, a, b, **kwargs)
        
        if interpType == "lagrange":
            _wnodes = {}
            for x in self.wnodes.keys():
                _wnodes[x] = self.lagrangeWeight(x)
            self.wnodes = _wnodes
        

    def I(self, f):
        return sum(f(q)*w for q,w in self.wnodes.items())
    

    def lagrangeBasis(self, x):
        "Returns lagrange interpolant with value 1 in $node"

        xGrid = self.getList()
        yGrid = [0]*len(self.wnodes)
        i = xGrid.index(x)
        yGrid[i] = 1
        return lagrange(xGrid, yGrid)

    
    def lagrangeWeight(self,x):
        "Returns the weight of the lagrange interpolant with value 1 in $x"
        return (self.lagrangeBasis(x).integ()(self.b)-
                self.lagrangeBasis(x).integ()(self.a))


    def legendreP(N):
        if N==0:
            return np.poly1d([1])
        if N==1:
            return np.poly1d([1,0])
        if N>1:
            return (((2*N+1) * np.poly1d([1,0]) * Quad.legendreP(N-1)
                     -N*Quad.legendreP(N-2))/(N+1))


    def getWList(self):
        return sorted(self.wnodes.values())


    def peanoArg(k,x,t):
        return posit(x-t)**k

    def peanoInt(k,t,a,b):
        if a<=b:
            if t>=b:
                return 0
            else:
                return (b-t)**(k+1) /(k+1)
        else:
            return -Quad.peanoInt(k,t,b,a)

    def peanoKer(self, k, t):
        return (Quad.peanoInt(k,t,self.a,self.b) -
                sum(Quad.peanoArg(k,node,t)*weight for
                    node,weight in self.wnodes.items()))

    
        
