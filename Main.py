#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:26:08 2026

@author: Christopher Wolczak 24983773
"""
# MAIN LIMITATIONS
# SIMPLY SUPPORTED BEAM ASSUMPTION
# SPLINE CAN OVERSHOOT BETWEEN POINTS

import numpy as np
import matplotlib.pyplot as plt
from Loads import rectangle,triangle,trapezoid,tributary
from Integration import integral, analyse

# inputs
w = 0.001 # Wind pressure, N/mm^2 or MPa
H = 1200 # Mullion height, mm
L = 1000 # Mullion tributary width, mm
E = 70000 # Elasticity Modulus, N/mm^2 or MPa
I = 2.06e6 # mm^4
x = np.linspace(0, H, 200) # Creates 200 points between 0 and H

a, w_peak = tributary(w,L,H) # Finds a and w_peak for 45 degree cases

y = rectangle(x,L,w) + trapezoid(x,H,a, w_peak) # Superimposes load cases

_ = analyse(x,y,H,E,I) # Analysis and returning Max Deflection + Bending Moment