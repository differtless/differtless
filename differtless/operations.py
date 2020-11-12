import numpy as np
import scipy as sp
from ad import FuncInput

'''
Re-defining numpy and scipy functions to return FuncInput objects of (value, gradient)
'''

'''
Numpy
'''

# Exponents and logarithms

def exp(x):
    return FuncInput(np.exp(x), np.exp(x))

def expm1(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def exp2(x):
    raise NotImplementedError('Function not yet implemented in differtless')

def log(x):
    return FuncInput(np.log(x), 1/x)

def log10(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def log1p(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def logaddexp(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def logaddexp2(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')
    
# Trigonometric functions

def sin(x):
    return FuncInput(np.sin(x), np.cos(x))

def cos(x):
    return FuncInput(np.cos(x), -np.sin(x))

def tan(x):
    return FuncInput(np.tan(x), 1/np.cos(x))

def arcsin(x):
    raise NotImplementedError('Function not yet implemented in differtless')

def arccos(x):
    raise NotImplementedError('Function not yet implemented in differtless')

def arctan(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def hypot(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def arctan2(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')
    
# Hyperbolic functions

def sinh(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def cosh(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def tanh(x):
    raise NotImplementedError('Function not yet implemented in differtless')

def arcsinh(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def arccosh(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
def arctanh(x):
    raise NotImplementedError('Function not yet implemented in differtless')
    
'''
Scipy
'''

# For both numpy and scipy we can implement functions like the PDFs of various distributions
# as these are common use-cases for AD (e.g. in for samplers like HMC or VI)