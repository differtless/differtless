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

# Wrapper that will make sure the input is either type FuncInput or a real number
def validate_input(func):
    def wrapper(x):
        if not isinstance(x, FuncInput) and not isinstance(x, numbers.Real):
            raise TypeError('Inputs must be type FuncInput or a real number')
        return func(x)
    return wrapper



# Exponents and logarithms
@validate_input
def exp(x):
    if isinstance(x, FuncInput):
        new_vals = np.exp(x.val_)
        new_ders = [x.ders_[i] * np.exp(x.val_) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.exp(x)

def expm1(x):
    return exp(x) - 1

def exp2(x):
    return 2**x

def sqrt(x):
    return x**0.5

@validate_input
def log(x):
    if isinstance(x, FuncInput):
        new_vals = np.log(x.val_)
        new_ders = [x.ders_[i] * (1/x.val_) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.log(x)

@validate_input
def log10(x):
    if isinstance(x, FuncInput):
        new_vals = np.log10(x.val_)
        new_ders = [x.ders_[i] * (1/(x.val_ * np.log(10))) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.log10(x)

@validate_input
def log2(x):
    if isinstance(x, FuncInput):
        new_vals = np.log2(x.val_)
        new_ders = [x.ders_[i] * (1/(x.val_ * np.log(2))) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
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
        new_ders = [x.ders_[i] * np.cos(x.val_) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.sin(x)

@validate_input
def cos(x):
    if isinstance(x, FuncInput):
        new_vals = np.cos(x.val_)
        new_ders = [x.ders_[i] * (-np.sin(x.val_)) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.cos(x)

def tan(x):
    if isinstance(x, FuncInput):
        new_vals = np.tan(x.val_)
        new_ders = [x.ders_[i] * (1/np.cos(x.val_))**2 for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.tan(x)

def arcsin(x):
    if isinstance(x, FuncInput):
        assert x.val_ > -1 and x.val_ < 1, 'Input is outside the domain of arcsin or its derivative'
        new_val = np.arcsin(x.val_)
        new_ders = [(1/sqrt(1 - x.val_)) * self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x > -1 and x < 1, 'Input is outside the domain of arcsin or its derivative'
        return np.arcsin(x)

def arccos(x):
    if isinstance(x, FuncInput):
        assert x.val_ > -1 and x.val_ < 1, 'Input is outside the domain of arccos or its derivative'
        new_val = np.arccos(x.val_)
        new_ders = [(-(1/sqrt(1 - x.val_))) * self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x > -1 and x < 1, 'Input is outside the domain of arccos or its derivative'
        return np.arccos(x)

def arctan(x):
    if isinstance(x, FuncInput):
        new_val = np.arctan(x.val_)
        new_ders = [(1/(1 + x.val_)) * self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.arctan(x)

def hypot(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')

def arctan2(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')

# Hyperbolic functions

def sinh(x):
    if isinstance(x, FuncInput):
        new_val = np.sinh(x.val_)
        new_ders = [np.cosh(x.val_) * self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.sinh(x)

def cosh(x):
    if isinstance(x, FuncInput):
        new_val = np.cosh(x.val_)
        new_ders = [(-np.sinh(x.val_)) * self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.cosh(x)

def tanh(x):
    if isinstance(x, FuncInput):
        new_val = np.tanh(x.val_)
        new_ders = [(np.sech(x.val_) ** 2) * self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.tanh(x)

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
