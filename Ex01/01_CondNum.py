import numpy as np
from matplotlib import pylab as plt

### Conditioning numbers

## Ex. 1.1
def FD(f,x,h):
    "returns the incremental difference of f in x with step h"
    return (f(x+h)-f(x))/h

## Ex. 1.2
FDSinList10 = [FD(np.sin, 1, 10**h) for h in range(-56,0)]
# a list of finite incremental differences of sin in 1 with decreasing step

FDErrorSinList10 = [abs(y - np.cos(1.0)) for y in FDSinList10]
# a list of errors in the finite approximation of sin' with decreasing steps

## Ex. 1.3
plt.loglog(FDErrorSinList10)

## Ex. 1.4
FDSinList2 = [FD(np.sin, 1, 2**h) for h in range(-56,0)]
FDErrorSinList2 = [abs(y - np.cos(1.0)) for y in FDSinList2]
plt.loglog(FDErrorSinList2)
# in this latter case we use base 2, which gives us a smaller minimum error;
# this is mainly due to machine representation of floating point numbers
# in base 2.

plt.title("Finite differences of sin with grid 10^(-i) vs 2^(-i)")
plt.show()

## Ex 1.5
def CFD(f,x,h):
    "returns the central incremental difference of f in x with step h"
    return (f(x+h)-f(x-h))/(2*h)

## Ex 1.6
CFDSinList10 = [CFD(np.sin, 1, 10**h) for h in range(-56,0)]
CFDErrorSinList10 = [abs(y - np.cos(1.0)) for y in CFDSinList10]
plt.plot(CFDErrorSinList10)

CFDSinList2 = [CFD(np.sin, 1, 2**h) for h in range(-56,0)]
CFDErrorSinList2 = [abs(y - np.cos(1.0)) for y in CFDSinList2]
plt.loglog(CFDErrorSinList2)

plt.title("Central finite differences of sin with grid 10^(-i) vs 2^(-i)")
plt.show()

# A quantitative evaluation of the error done with Finite differences vs
# central finite differences can be obtained applying Taylor theorem;
# in fact for finite differences one immediately sees (by definition) that
# the error decreases as the first derivative of the function, while for
# CFD (summing the taylor expansion with lagrange remainder to the 2nd order
# for f(x+h) and f(x-h)) one obtains that the error decreases as the f''.


## Ex 1.7
print("Machine eps: ", 3*(4./3 - 1) - 1)
# This is a good approximation of machine precision for the following reason:
# in every base  b  the division algorithm gives the periodic number
#       (b+2)/(b+1) = 1.0[b-1]0[b-1]0 ... .
# Truncating the number after n decimal digits (this is what
# the machine does), and subtracting 1 we obtain a number of the type
#      0.0[b-1]0[b-1]0 ...
# (but this time with only a finite number of digits).
# Multiplying now by  b+1  we obtain
#      0.[b-1][b-1][b-1] ...
# i.e. the closest number to 1 in machine representation;
# therefore by subtracting 1 we get the smallest representable error.

## Ex 1.8
import random

def getRandomList(n):
    "returns a list of n random numbers in [0,1]"
    return [random.random() for i in range(n)]


def sumList32(list_):
    "returns the sum of the elements of list_ with 32bit precision"
    accumulator = 0
    for i in range(len(list_)):
        accumulator = np.float32(accumulator + list_[i])
    return accumulator

def sumList16(list_):
    "returns the sum of the elements of list_ with 16bit precision"
    accumulator = 0
    for i in range(len(list_)):
        accumulator = np.float16(accumulator + list_[i])
    return accumulator

def kahan(list_):
    "returns the sum of the elements of list_ using Kahan algorithm"
    s = list_[0]
    c = 0
    for i in range(1,len(list_)):
        y = list_[i]-c
        t = s+y
        c = (t-s) - y
        s = t
    return s
# the Kahan algorithm is more accurate than mindless sorting since
# it preserves in c the eventually small difference between y and s,
# which would be otherwise cancelled due to machine precision

# some examples for a list of 10^6 uniformly distributed numbers between [0,1]:
l = getRandomList(10**6)
print("Stock sum: ", sum(l))
print("Sorted stock sum: ", sum(sorted(l[:])))
print("Reversed sorted stock sum: ", sum(l[::1]))
print("16bit sum: ", sumList16(l))
print("32bit sum: ", sumList32(l))
print("Sorted 32bit sum: ", sumList32(sorted(l[:])))
print("Reversed sorted 32bit sum: ", sumList32(l[::1]))
print("Kahan algorithm: ", kahan(l))
