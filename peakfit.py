# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: py3 venv
#     language: python
#     name: py3
# ---

import numpy as np
from scipy.optimize import curve_fit


# # Fit Peak

class Gauss:
    """Gaussian function"""
    def __init__(self, x0=None, fwhm=None, ampl=None):
        self.x0_init = x0
        self.fwhm_init = fwhm
        self.amplitude_init = ampl
        self.param_name = [('x0', 'fwhm', 'amplitude'), ]
        self.name = 'Gaussian'
        
    def __call__(self, x, x0, fwhm, amplitude):
        sigma = fwhm /( 2*np.sqrt(2*np.log(2)) )
        return amplitude * np.exp( -(x-x0)**2/(2*sigma**2) )
    
    def estimate_param(self, x, y):
        if not self.x0_init:
            self.x0_init = x[np.argmax(y)]
        if not self.fwhm_init:
            self.fwhm_init = np.ptp( x[ y  > (y.min() + y.max())/2 ] )
        if not self.amplitude_init:
            self.amplitude_init = np.ptp(y)
 
        return self.x0_init, self.fwhm_init, self.amplitude_init


class Linear:
    """Linear function"""
    def __init__(self, slope=None, intercept=None):
        self.slope_init = slope
        self.intercept_init = intercept
        self.param_name = [('slope', 'intercept'), ]
        self.name = 'linear'
        
    def __call__(self, x, slope, intercept):
        return x*slope + intercept
    
    def estimate_param(self, x, y):
        if not self.slope_init:
            self.slope_init = (y[-1] - y[0])/(x[-1] - x[0])
        if not self.intercept_init:
            self.intercept_init = y[0] - self.slope_init*x[0]
 
        return self.slope_init, self.intercept_init


class Lorentzian:
    """Lorentzian function (or Cauchy distribution)
        
        I = 1/( 1 + x^2 )
    """
    def __init__(self, x0=None, fwhm=None, ampl=None):
        self.x0_init = x0
        self.fwhm_init = fwhm
        self.amplitude_init = ampl
        self.param_name = [('x0', 'fwhm', 'amplitude'), ]
        self.name = 'Lorentzian'
        
    def __call__(self, x, x0, fwhm, amplitude):
        hwhm = fwhm / 2
        u = x - x0
        return amplitude/( 1 + (u/hwhm)**2 )
    
    def estimate_param(self, x, y):
        if not self.x0_init:
            self.x0_init = x[np.argmax(y)]
        if not self.fwhm_init:
            self.fwhm_init = np.ptp( x[ y  > (y.min() + y.max())/2 ] )
        if not self.amplitude_init:
            self.amplitude_init = np.ptp(y)
 
        return self.x0_init, self.fwhm_init, self.amplitude_init


class Sum:
    """Build a new function as the sum of two functions"""
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
        self.param_name = a.param_name + b.param_name
        
    def __call__(self, x, *p):
        nargs_a = sum(len(u) for u in self.a.param_name)
        p_a = p[:nargs_a]
        p_b = p[nargs_a:]
        return self.a(x, *p_a) + self.b(x, *p_b)
    
    def estimate_param(self, x, y):
        p_a = self.a.estimate_param(x, y)
        p_b = self.b.estimate_param(x, y)
        return (*p_a, *p_b)


def peakfit(x, y, function=Gauss(), background=Linear()):
    """Fit the data (x, y
        using the provided function)"""

    if background is not None:
        function = Sum(function, background)
        
    p0 = function.estimate_param(x, y)
 
    popt, pcov = curve_fit(function, x, y, p0)

    result = []
    idx = 0
    for names in function.param_name:
        res = {}
        for name in names:
            res[name] = popt[idx]
            idx += 1

        result.append(res)
    
    return result, lambda x:function(x, *popt)

# +



