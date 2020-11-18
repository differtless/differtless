import numpy as np
import numbers
from ad import FuncInput

'''
Re-defining numpy and scipy functions to return FuncInput objects of (value, gradient)
'''

'''
Numpy
'''

# Wrapper that will make sure the input is either type FuncInput or a real number
def validate_input(func):
    def wrapper(self):
        if not isinstance(self, FuncInput) and not isinstance(self, numbers.Real):
            raise TypeError('Inputs must be type FuncInput or a real number')
        return func(self)
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
    return exp(x) - 1

def exp2(x):
    return 2**x

@validate_input
def log(x):
    if isinstance(x, FuncInput):
        new_vals = np.log(x.val_)
        new_ders = x.ders_ * (1/x.val_)
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.log(x)

@validate_input
def log10(x):
    if isinstance(x, FuncInput):
        new_vals = np.log10(x.val_)
        new_ders = x.ders_ * (1/(x.val_ * np.log(10)))
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.log10(x)

@validate_input
def log2(x):
    if isinstance(x, FuncInput):
        new_vals = np.log2(x.val_)
        new_ders = x.ders_ * (1/(x.val_ * np.log(2)))
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.log2(x)


@validate_input
def log1p(x):
    return log(1 + x)

def logaddexp(x1, x2):
    return log(exp(x1) + exp(x2))

def logaddexp2(x1, x2):
    return log2(x1**2 + x2**2)

# Trigonometric functions
@validate_input
def sin(x):
    if isinstance(x, FuncInput):
        new_vals = np.sin(x.val_)
        new_ders = x.ders_ * np.cos(x.val_)
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.sin(x)

@validate_input
def cos(x):
    if isinstance(x, FuncInput):
        new_vals = np.cos(x.val_)
        new_ders = x.ders_ * (-np.sin(x.val_))
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.cos(x)

def tan(x):
    if isinstance(x, FuncInput):
        new_vals = np.tan(x.val_)
        new_ders = x.ders_ * (1/np.cos(x.val_))**2
        return FuncInput(new_vals, new_ders)
    elif isinput(x, numbers.Real):
        return np.tan(x)

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