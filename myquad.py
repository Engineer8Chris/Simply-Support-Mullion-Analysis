# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:08:39 2022

@author: Chris
Set of functions for numerical quadrature in 1D
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import p_roots

def qtrapn(f,a,b,N):
    # perform trapezoidal rule on a 1D function f, 
    # on the interval [a,b], with N subintervals
    
    h = (b-a)/N
    
    xj = np.array([a+h*j for j in range(N+1)])

    I = h/2*(f(xj[0]) + 2*sum(f(xj[1:-1])) + f(xj[-1]))
    
    return I

def qsimpn(f,a,b,N):
    # perform simpson's composite rule on a 1D function f, 
    # on the interval [a,b], with N subintervals
    
    h = (b-a)/N
    
    xj = np.array([a+h*j for j in range(N+1)])

    I = h/3*(f(xj[0]) + 4*sum(f(xj[1:-1:2]))
                      + 2*sum(f(xj[2:-1:2]))
                      + f(xj[-1]))
    
    return I


def qtrapz(f,a,b,tol):
    # "fast" refined trapedzoidal rule
    maxp = 100
    
    
    h = (b-a)
    
    I0 = h/2*(f(a) + f(b)) # first trapezoidal step
    
  
    for p in range(maxp):
           
        xj = np.array([a+h/2+h*j for j in range(2**p)])
        
        I = I0/2 + h/2*sum(f(xj))
       
        if p>3 and abs(I-I0)<tol:
            return I
        h = h/2    
        I0 = I
        
    print('qtrapz: quadrature did not achieve specified tolerance.')
    return float('nan')

def quadz(f,a,b,tol):
    # do Gauss-Legendre quadrature
    # on a 1D function f with the interval [a,b]
    # and required tolerance tol
    #
    NMAX = 100  #maximum number of points for quadrature
    Iold = 0
    
    for n in range(1,NMAX):
        # set up zeros and weights
        [xj,wj]=p_roots(n)
    
        #I = (b-a)/2*sum(wj*f((b-a)/2*xj+(b+a)/2))
        W = [(b-a)/2*(wj[i]*f((b-a)/2*xj[i]+(b+a)/2)) for i in range(len(xj))]
        I = sum(W)
        #print(n,I)
        
        if n>3 and abs(I-Iold)<tol:
            return I
        
        Iold = I
        
    print('quadz: Quadrature did not converge to required tolerance.')
    return float("NaN")


