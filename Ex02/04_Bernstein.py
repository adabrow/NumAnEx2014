### Bernstein Basis
import numpy as np


## Ex 4.1
from scipy.special import binom

def B1(i,n,t):
    """Restituisce il polinomio di Bernstein (i,n) valutato in t,
    usando la definizione binomiale"""
    if i < 0 or i > n:
        return 0
    return binom(n,i)* t**i * (1-t)**(n-i)

def B2(i,n,t):
    """Restituisce il polinomio di Bernstein (i,n) valutato in t,
    usando una definizione ricorsiva"""
    if i < 0 or i > n:
        return 0
    elif n == 0:
        return 1

    return (1-t) * B2(i,n-1,t) + t * B2(i-1,n-1,t)

def B3(k,n,t):
    """Restituisce il polinomio di Bernstein (i,n) valutato in t,
    usando l'espressione in serie di potenze"""
    return sum([(-1)**(i-k) * binom(n,i) * binom(i,k) * t**i
         for i in range(k, n+1)])


## Ex 4.2
import cProfile

cProfile.run("B1(7,20,1)")
cProfile.run("B2(7,20,1)")
cProfile.run("B3(7,20,1)")


## Ex 4.3
from matplotlib import pylab as plt

# Segue: plot di B1(i, n, x) per n in [20,40,80,160,320].
step = 2**(-7)
plotNodes = np.arange(0, 1+step, step)
for n in 10 * (2**np.arange(1,6,1)):
    for i in range(n+1):
        plt.plot(plotNodes, [B1(i,n,x) for x in plotNodes])
    plt.title("B1(i, "+ str(n) + ", x)")
    plt.show()

# Segue: codice per il plot di B2.
# ! E' stato commentato in quanto l'algoritmo ricorsivo impiega un 
# tempo eccessivo anche solo per calcolare la griglia per n = 20.

#plotNodes = np.arange(0, 1+step, step)
#for n in 10 * (2**np.arange(1,6,1)):
#    for i in range(n+1):
#       plt.plot(plotNodes, [B2(i,n,x) for x in plotNodes])
#    plt.title("B2(i, "+ str(n) + ", x)")
#    plt.show()

# Segue: codice per il plot di B3
# ! Per ridurre il tempo di esecuzione l'algoritmo calcola B3(i, n, x)
# solo per n = 20, 40.
plotNodes = np.arange(0, 1+step, step)
for n in 10 * (2**np.arange(1,3,1)):
    for i in range(n+1):
        plt.plot(plotNodes, [B3(i,n,x) for x in plotNodes])
    plt.title("B3(i, "+ str(n) + ", x)")
    plt.show()
# Nel grafico per n = 40 si possono già notare gravi errori
# di approssimazione nelle vicinanze di 1, dovuti alla presenza
# di molte somme di termini molto piccoli.


## Ex 4.4
# Segue una reimplementazione della definizione di B3 con il modulo
# scipy.polynomial.
import numpy.polynomial as P

def B3Poly(i,n):
    """Restituisce il polinomio di Bernstein (i,n), implementato con
    numpy.polynomial e usando l'espressione in serie di potenze"""
    outPoly = P.Polynomial([0])
    for k in range(i,n+1):
        nextTerm = k*[0]
        nextTerm.append((-1)**(i-k) * binom(n,i) * binom(i,k))
        outPoly += nextTerm
    return outPoly

def DB3Poly(i,n):
    """Restituisce la derivata del polinomio di Bernstein (i,n),
    implementato con numpy.polynomial e usando l'espressione in serie
    di potenze"""
    return B3Poly(i,n).deriv()

# Tuttavia questa implementazione ha ancora problemi di stabilità
# numerica non risolti dalla rappresentazione con numpy.polynomial.
# Svolgendo i conti a mano per la derivata di B1 otteniamo la
# seguente funzione, numericamente più stabile.

def DB1(i,n,x):
    return n * (B1(i-1,n-1,x) - B1(i,n-1,x))
    

## Ex 4.5
def B(n,f,x):
    return sum([B1(i,n,x) * f(i/n) for i in range(n+1)])

for j in 4**np.arange(1,5,1):
    plt.plot(plotNodes,
             [B(j, lambda y: np.cos(16*y), x)
              for x in plotNodes])
    plt.title(str(j) + "th Bernstein approximation of cos(16*x)")
    plt.show()
