import numpy as np
import scipy as sp
import numbers
from ad import FuncInput

'''
Re-defining numpy and scipy functions to return FuncInput objects of (value, gradient)
'''

'''
Numpy
'''

# Wrapper that will make sure certain specifications are met for the inputs
def validate_input(func):
    def wrapper(self):
        if not isinstance(self, FuncInput) or not isinstance(self, numbers.Real):
            raise TypeError('Inputs must be type FuncInput or a real number')
        return func(self, other)
    return wrapper


# Exponents and logarithms
@validate_input
def exp(x):
    if isinstance(x, FuncInput):
        new_vals = np.exp(x.val_)
        new_ders = x.ders_ * np.exp(x.val_)
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.exp(x)

def expm1(x):
    raise NotImplementedError('Function not yet implemented in differtless')

def exp2(x):
    raise NotImplementedError('Function not yet implemented in differtless')

@validate_input
def log(x):
    if isinstance(x, FuncInput):
        new_vals = np.log(x.val_)
        new_ders = x.ders_ * (1/x.val_
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real)
        return np.log(x)

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
