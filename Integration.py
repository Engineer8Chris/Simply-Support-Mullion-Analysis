#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:48:22 2026

@author: Christopher Wolczak 24983773
"""
import numpy as np
import matplotlib.pyplot as plt
from myquad import qtrapz

def integral(x,y, rtol = 1e-6):
    
    f = lambda t: np.interp(t, x, y) #Converts 'load array' into a function
    
    tol = rtol * np.abs(y).max() * (x[-1]-x[0])  # Tolerance scaled to the size of the problem
    
    result = np.zeros_like(y) # Empty array for the running integral (starts at 0)
    
    for i in range(1,len(x)):
        
        result[i] = qtrapz(f, x[0], x[i], tol) # Refined trapezoidal rule from the start up to x[i]
        
    return result


def analyse(x, y, H, E, I):
    
    M_B = integral(x, y * (H-x))[-1] # Moment of the whole load about support B (at x = H)

    R_A = M_B / H # Left reaction, from moments about B: R_A * H = M_B

    S = R_A - integral(x, y, rtol = 1e-6) # Shear Force - reaction minus the load so far

    BM = integral(x, S, rtol = 1e-6) # Bending Moment - integral of shear

    SL = -integral(x,(BM/(E*I)),rtol = 1e-6) # Slope - integral of M/EI (negative so windward deflection is positive)

    DFL = integral(x, SL, rtol = 1e-6) # Deflection

    C1  = -DFL[-1] / H          # Chosen so that deflection at x = H is 0
    DFL = DFL + C1 * x          # Corrected deflection (zero at both supports)
    SL  = SL + C1               # The slope needs the same constant added
    
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8,8))

    ax[0].plot(x, y)
    ax[0].set_ylabel("Load (N/mm)")
    ax[0].set_xlabel("Position along mullion (mm)")

    ax[1].plot(x, S)
    ax[1].set_ylabel("Shear force (N)")
    ax[1].set_xlabel("Position along mullion (mm)")

    ax[2].plot(x, BM)
    ax[2].set_ylabel("Bending Moment (Nmm)")
    ax[2].set_xlabel("Position along mullion (mm)")

    ax[3].plot(x, DFL)
    ax[3].set_ylabel("Deflection (mm)")
    ax[3].set_xlabel("Position along mullion (mm)")
    
    r = y.argmax()
    i = DFL.argmax()
    z = BM.argmax()
    q = S.argmax()
    
    print("Max Loading:", y[r], "N/mm at x =", x[r], "mm")
    print("Max Shear:", S[q], "N mm at x =", x[q], "mm")
    print("Max Bending Moment:", BM[z], "N mm at x =", x[z], "mm")
    print("Max Deflection:", DFL[i], "mm at x =", x[i], "mm")
  
    return R_A, S, BM, SL, DFL # Reaction and the full diagrams for use in Main


    
    
    
    
    
    
    
    
    
    
    
