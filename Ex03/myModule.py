import numpy as np
from numpy.polynomial.chebyshev import chebgauss
from matplotlib import pylab as plt
from scipy.interpolate import lagrange
from numpy.polynomial.legendre import leggauss

def posit(x):
        if x < 0: return 0
        if x >=0: return x

class _Nodes:
    """Saves %wnodes in the keys of a dictionary $wnodes (default values are
    None). $a, $b are expected to be the extrema of the interval in which the
    nodes are in. Has many optional arguments (see further doc.).


    Represents nodes as keys of a dictionary (in the subclass WNodes the values
    become weights associated to the node). Being this only a blueprint for
    WNodes, it is advised to use the latter.
    
    __init__ handles arguments with the following behavior:
    $wnodes is expected to be a dictionary - in which case is copied into a 
    variable - or an iterable - is converted into a dictionary with uniform
    values None - or another _Nodes object - keys are copied, values set None;
    $a, $b should be the extrema of the interval in which all nodes are in;
    %equi is an optional argument used to generate equispaced nodes in [a,b] -
    it is expected to be a dictionary containing a key nNodes with value
    the number of nodes, and an optional key extrIncl with value True or False
    depending if extrema should be or should not included in the nodes.
    $cheb is an optional argument used to generate Chebyshev nodes in [a,b],
    which is expected to be a dictionary, which works as in $equi.
    
    The class contains also some "static" methods for general purpose nodes
    generation: eqNodes - which return a list of N equispaced nodes in [a,b] -
    and chebNodes - which returns a list of N Chebyshev nodes in [a,b].
    """
    
    def __init__(self, wnodes={}, a=0, b=1, **kwargs):
        
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
        """return a list of N equispaced nodes in [a,b].
        If extrIncl is True, a,b are included, otherwise not."""

        if extrIncl:
            return np.linspace(a,b,N)
        else:
            out = np.linspace(a,b,N+2)
            return list(np.array_split(out,[1,len(out)-1])[1])
            # subdivides an array in subarrays containing respectively
            # a,center,b and returns center (i.e. a linspace without extrema).


    def chebNodes(N, a=0, b=1, extrIncl=False):
        """returns a list of N Chebyshev nodes in [a,b].
        If extrIncl is True, a,b are included, otherwise not."""
    
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
        "Returns a sorted list of nodes."
        return sorted(self.wnodes)
            
            
class WNodes(_Nodes):
    """Saves $wnodes in the keys of a dictionary $wnodes. Constructor arguments
    behave as in _Nodes, but have also additional options (see further doc.).


    Subclass of _Nodes, represents nodes and weights respectively as keys and
    values in a single dictionary $wnodes.
    
    __init__ handles arguments as in _Nodes, with the addition of the optional
    argument %weights - when %weights is passed, the constructor tries to
    associate to the nodes the weights specified in $weights.

    The class contains also the methods iterWNodes - which rescales nodes and
    weights by a factor 1/N - and the utility method uniformWeights - which
    sets all weights associated to current nodes to %val."""

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


    def iterWNodes(self, N):
        "Rescales $wnodes by a factor 1/N"
        a, b = self.a, self.b
        if a in self.wnodes.keys() and b in self.wnodes.keys():
        # We have to check if both extrema are included: in this case the weights
        # in the images (through rescaling and translation) of the extrema inside
        # (a,b) are the sum of the weights of the extrema.

            weightA, weightB = self.wnodes[a], self.wnodes[b]
                
            # translate into 0 the internal nodes and rescale them by 1/N
            _wnodes = {(q-a)/N : w/N for q,w in self.wnodes.items()
                       if q!=a and q!=b}
                
            # scale back into [a,b] internal nodes
            self.wnodes = {q + j*(b-a)/N +a: w for j in range(N)
                           for q,w in _wnodes.items()}
                
            # rescale weights also in the extrema
            self.wnodes.update({a : weightA/N, b : weightB/N})
                
            # sum both weights in the nodes in (a,b) which, through rescaling,
            # are image of a, b.
            for j in range(1,N):
                self.wnodes.update({a + j*(b-a)/N : (weightA + weightB)/N})
        
        else:
            _wnodes = {(q-a)/N : w/N for q,w in self.wnodes.items()}
            self.wnodes = {q + j*(b-a)/N +a : w for j in range(N)
                           for q,w in _wnodes.items()}



    def uniformWeights(self, val):
        "Sets all weights to $val."
        for x in self.wnodes.keys():
            self.wnodes[x] = val
    


class Quad(WNodes):
    """Constructor arguments behaves as in WNodes, but are available additional
    options (see further doc.).


    Subclass of WNodes, it is useful for weights calculations using specified
    rules (e.g. Newton-Cotes) and quadrature of funcions.
    
    __init__ handles arguments as in _WNodes, with the following additions:
    if the weightsCalc argument is passed with value val, the constructor
    calculates weights from nodes with rule val (currently supports only
    "lagrange", which uses lagrange interpolants);
    if the newtonCotes argument is passed, its value is expected to be a
    dictionary containing the keys: %degree - which specifies the degree of the
    rule - and %type - which can have value 'open' or 'closed'.

    There are different additional methods:
    I, given a function, returns its quadrature computed at $wnodes;
    lagrangeWeight returns the definite integral from $a to $b of a lagrange
    basis polynomial;
    the methods lagrangeBasis, lagrangeWeights, calcWeightsLagrange, calculate
    lagrange interpolants, integrate them and use them to get the weights
    starting from known nodes;
    the "static" method legendreP returns the %N-th Legendre polynomial;
    the "static" methods peanoArg, peanoInt return resp. the %k-th Peano 
    argument function and its integral from $a to $b;
    peanoKer returns the %k-th Peano kernel on [$a, $b] evaluated in %t.
    """

    def __init__(self, wnodes={}, a=0, b=1, weightsCalc=None,
                 iterations=0, **kwargs):

        if "newtonCotes" in kwargs:
            n = kwargs["newtonCotes"]["degree"]
            
            if "type" in kwargs["newtonCotes"].keys():
                if kwargs["newtonCotes"]["type"] == "open":
                    t = False
                    n -= 1
                else:
                    t = True
                    n += 1
            else: # defaults to closed Newton-Cotes
                t = True
                n += 1
                
            _kwargs = kwargs.copy()
            del _kwargs["newtonCotes"]
            _kwargs["equi"] = {"nNodes":n, "extrIncl":t}
            super().__init__(wnodes, a, b, **_kwargs)

        else:
            super().__init__(wnodes, a, b, **kwargs)
        
        if weightsCalc == "lagrange":
            self.calcWeightsLagrange()
            

    def I(self, f):
        "Returns the quadrature of the function f in $wnodes"
        return sum(f(q)*w for q,w in self.wnodes.items())
    

    def lagrangeBasis(self, x):
        "Returns lagrange interpolant which has value 1 in x"

        xGrid = self.getList()
        yGrid = [0]*len(self.wnodes)
        i = xGrid.index(x)
        yGrid[i] = 1
        return lagrange(xGrid, yGrid)

    
    def lagrangeWeight(self,x):
        "Returns the weight of the lagrange interpolant which has value 1 in x"
        return (self.lagrangeBasis(x).integ()(self.b)-
                self.lagrangeBasis(x).integ()(self.a))


    def calcWeightsLagrange(self):
        "Calculates the weights associated to the current nodes in $wnodes."
        _wnodes = {}
        for x in self.wnodes.keys():
            _wnodes[x] = self.lagrangeWeight(x)
        self.wnodes = _wnodes


    def legendreP(N):
        "Returns the N-th Legendre polynomial."
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
        "Returns the k-th Peano argument function evaluated at x-t."
        return posit(x-t)**k

    def peanoInt(k,t,a,b):
        "Returns the integral from a to b of the k-th Peano argument function."
        if a<=b:
            if t>=b:
                return 0
            else:
                return (b-t)**(k+1) /(k+1)
        else:
            return -Quad.peanoInt(k,t,b,a)

    def peanoKer(self, k, t):
        "Returns the k-th Peano kernel on [$a, $b] evaluated in t."
        peanoQuad = sum(Quad.peanoArg(k,node,t)*weight for
                    node,weight in self.wnodes.items())
        return (Quad.peanoInt(k,t,self.a,self.b) - peanoQuad)    
        
