import numpy as np
from scipy import special
import numbers
import warnings
from differtless.ad import FuncInput

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


@validate_input
def erf(x):
    if isinstance(x, FuncInput):
        new_vals = special.erf(x.val_)
        new_ders = [x.ders_[i] * 2/(np.pi**0.5) * np.exp(- (x.val_)**2) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return special.erf(x)


@validate_input
def gamma(x):
    if isinstance(x, FuncInput):
        new_vals = special.gamma(x.val_)
        new_ders = [x.ders_[i] * 2/(np.pi**0.5) * np.exp(special.digamma(x.val_)) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return special.gamma(x)


def factorial(x):
    return gamma(x+1)


@validate_input
def floor(x):
    if isinstance(x, FuncInput):
        new_vals = np.floor(x.val_)
        new_ders = [x.ders_[i] * 0 for i in range(len(x.ders_))] # technically not defined at non-integers
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.floor(x)


class Normal():
    
    def __init__(self, loc=0, scale=1):
        '''
        Normal distribution with mean `loc` and standard deviation `scale`
        '''
        self.loc = loc
        self.scale = scale
        
    def __str__(self):
        return f'Normal distribution with mean {self.loc} and standard deviation {self.scale}'
    
    def __repr__(self):
        return f'Normal(loc={self.loc}, scale={self.scale})'
    
    def pdf(self, x):
        return 1/(self.scale * (2*np.pi)**0.5) * exp(-0.5 * ((x - self.loc)/self.scale)**2)
    
    def logpdf(self, x):
        return -log(self.scale * (2*np.pi)**0.5) - 0.5* ((x - self.loc)/self.scale)**2
    
    def cdf(self, x):
        return 0.5*(1 + erf((x-self.loc)/(self.scale * 2**0.5)))
    
    def logcdf(self, x):
        return log(0.5) + log(1 + erf((x-self.loc)/(self.scale * 2**0.5)))


class Poisson():
    
    def __init__(self, mu):
        '''
        Poisson distribution with shape parameter `mu`
        '''
        self.mu = mu
        
    def __str__(self):
        return f'Poisson distribution with shape parameter {self.mu}'
    
    def __repr__(self):
        return f'Poisson(loc={self.loc}, scale={self.scale})'
    
    def pmf(self, x):
        return exp(-self.mu)*(self.mu**x)/(factorial(x))
    
    def logpmf(self, x):
        return -self.mu + x*log(self.mu) - log(factorial(x))
    
    def cdf(self, x):
        if isinstance(x, numbers.Real):
            return exp(-self.mu)*sum([self.mu**i / factorial(i) for i in range(int(floor(x))+1)])
        elif isinstance(x, FuncInput):
            warnings.warn('Cannot provide derivative for CDF of a discrete distribution – \
please try using finite differences. Returning only CDF value instead of FuncInput object...')
            return exp(-self.mu)*sum([self.mu**i / factorial(i) for i in range(int(floor(x.val_[0]))+1)])
    
    def logcdf(self, x):
        if isinstance(x, numbers.Real):
            return -self.mu + log(sum([self.mu**i / factorial(i) for i in range(int(floor(x))+1)]))
        elif isinstance(x, FuncInput):
            warnings.warn('Cannot provide derivative for CDF of a discrete distribution – \
please try using finite differences. Returning only CDF value instead of FuncInput object...')
            return -self.mu + log(sum([self.mu**i / factorial(i) for i in range(int(floor(x.val_[0]))+1)]))