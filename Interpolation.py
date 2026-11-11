# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:04:08 2026

@author: Christopher Wolczak 24983773
"""
import numpy as np

def natural_spline(xd,yd):
    xd = np.asarray(xd, dtype=float) #Covert to numpy array to index with array later
    yd = np.asarray(yd, dtype=float) #Covert to numpy array to index with array later
    
    n = len(xd) # Number of data points
    h = np.diff(xd) # Measures gaps for each cubic
    
    A = np.zeros((n,n)) # Create n x n Matrix
    r = np.zeros(n) # Create n Length Vector
    
    A[0,0] = 1 # Natural end - 1*M[0] = 0 so 0 curvature at the start
    A[-1,-1] = 1 # Natural end - 1*M[n-1] = 0 so 0 curvature at the end
    
    for i in range(1,n - 1):
        A[i, i-1] = h[i-1] # Sets the left hand side curvature
        A[i, i] = 2 * (h[i-1] + h[i]) #Sets the middle curvature
        A[i, i+1] = h[i] #Sets the right hand side curvature
        r[i] = 6 * ((yd[i+1] - yd[i])/ h[i] - (yd[i] - yd[i-1]) / h[i-1]) # 6 times (slope of the right gap - slope of the left gap)
    
    M = np.linalg.solve(A,r) # Finds all the curvatures at once
    
    def S(x):
        x = np.asarray(x, dtype=float) # Let x be a single number or array
        
        i = np.searchsorted(xd,x, side='right') - 1 # Find which gap each x sits in 
        
        i = np.clip(i, 0, n - 2) # Keep i a valid gap number so the last point doesn't fall off at the end
        
        hi = h[i] # Width of gap i.e. x_(j+1) - x_j
        
        A_w = (xd[i+1] - x)/hi      # A = (x_(j+1) - x)/(x_(j+1) - x_j)
        
        B_w = 1 - A_w               # B = 1 - A
        
        C = (A_w**3 - A_w) * hi**2 / 6 # C = (1/6)(A^3 - A)(x_(j+1) - x_j)^2
        
        D = (B_w**3 - B_w) * hi**2 / 6 # D = (1/6)(B^3 - B)(x_(j+1) - x_j)^2
     
        return A_w * yd[i] + B_w * yd[i+1] + C * M[i] + D * M[i+1] # y = A*y_j + B*y_(j+1) + C*q_j + D*q_(j+1)
    
    
    return S

