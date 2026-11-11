# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:43:13 2026

@author: Christopher Wolczak 24983773
"""
import numpy as np
from Interpolation import natural_spline

np.set_printoptions(suppress=True) # Get rid of the e when printing

def tributary (w, panel_width, panel_height): # Converts wind pressure on a panel into the peak line load on the mullion (45 degree rule)
    
    s = min(panel_width, panel_height) # Short side of the panel controls the load
    
    a = s/2  # Length over which the load rises from zero to its peak
    
    w_peak = w*s / 2 # Peak line load = pressure x widest strip feeding the mullion
    
    return a, w_peak

def rectangle (x, L, w): # Uniform line load along the mullion

    return w * L * np.ones_like(x)  #Pressure x width, same value at every point


def triangle (x,H,w_peak): # Triangular line load - zero at both ends with peak at midspan
    
    rising = (w_peak * x) / (H/2) # Straight line up from 0 at x = 0 to w_peak at x = H/2
    
    falling = (w_peak * (H - x)) / (H/2) # Straight line down from w_peak at x = H/2 to 0 at x = H
    
    return np.where(x < H/2, rising, falling) # Use rising in the first half, falling in the second

def trapezoid (x, H, a, w_peak): # Trapezoidal line load - rises over length a, flat in the middle and then falls over length a
    
    result = np.zeros_like(x) # Empty load array, same length as x
    
    for i in range(len(x)): # For each point in x
        
        if x[i] < a:
            result[i] = w_peak * x[i] / a # rising
            
        elif x[i] > H - a:
            result[i] = w_peak * (H - x[i]) / a # falling
            
        else:
            result[i] = w_peak # flat
   
    
    return result

def custom_load(x, heights, pressures, width):  # Line load from any measured pressure profile, using the cubic spline
    
    S = natural_spline(heights, pressures) # Smooth curve through the pressure readings
    
    p = np.maximum(S(x),0) # Pressure at every x, with any dip below zero set to 0
    
    return p * width # Pressure x width = line load (N/mm)


