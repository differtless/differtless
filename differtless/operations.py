import numpy as np
from scipy import special
from scipy.spatial import distance
import numbers
import warnings
from differtless.ad import FuncInput

'''
Re-defining numpy and scipy functions to return FuncInput objects of (value, gradient),
as well as various other useful functions and statistical distributions.
'''

# Wrapper that will make sure the input is either type FuncInput or a real number
def validate_input(func):
    def wrapper(x):
        if not isinstance(x, FuncInput) and not isinstance(x, numbers.Real):
            raise TypeError('Inputs must be type FuncInput or a real number')
        return func(x)
    return wrapper

'''
Numpy
'''

# Exponents and logarithms
@validate_input
def exp(x):
    """
    Returns the exponential of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on exp(x) and gradients based on exp'(x) = exp(x) * x

    Examples
    =======
    >>> x = FuncInput(np.array([0]),np.array([1,0]))
    >>> f = op.exp(x)
    >>> f
    FuncInput([1], [1, 0])
    """
    if isinstance(x, FuncInput):
        new_vals = np.exp(x.val_)
        new_ders = [x.ders_[i] * np.exp(x.val_) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.exp(x)

def expm1(x):
    """
    Returns the exponential minus 1 of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on exp(x)-1 and gradients based on exp'(x) = exp(x) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([0]),np.array([1,0]))
    >>> f = op.expm1(x)
    >>> f
    FuncInput([0], [1, 0])
    """
    return exp(x) - 1

def exp2(x):
    """
    Returns the 2 to the power of x of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on exp(x) and gradients based on 2**x * log(2) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([2]),np.array([1,0]))
    >>> f = op.exp2(x)
    >>> f
    FuncInput([4], [2.77258872, 0])
    """
    return 2**x

def expn(x, n): # exponential with base n
    return n**x

def sqrt(x):
    """
    Returns the square root of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on sqrt(x) and gradients based on 1/(2*sqrt(x)) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([4]),np.array([1,0]))
    >>> f = op.sqrt(x)
    >>> f
    FuncInput([2], [0.25, 0])
    """
    return x**0.5

@validate_input
def log(x):
    """
    Returns the natural log of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on log(x) and gradients based on log'(x) = 1/x * x'

    Examples
    =======
    >>> x = FuncInput(np.array([3]),np.array([1,0]))
    >>> f = op.log(x)
    >>> f
    FuncInput([1.09861229], [0.33333333,0])
    """
    if isinstance(x, FuncInput):
        new_vals = np.log(x.val_)
        new_ders = [x.ders_[i] * (1/x.val_) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.log(x)

@validate_input
def log10(x):
    """
    Returns the log of FuncInput object with the base of 10.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on log10(x) and gradients based on 1/(x*log(10)) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([3]),np.array([1,0]))
    >>> f = op.log10(x)
    >>> f
    FuncInput([0.47712125], [0.14476483,0])
    """
    if isinstance(x, FuncInput):
        new_vals = np.log10(x.val_)
        new_ders = [x.ders_[i] * (1/(x.val_ * np.log(10))) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.log10(x)

@validate_input
def log2(x):
    """
    Returns the log of FuncInput object with the base of 2.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on log2(x) and gradients based on 1/(x*log(2)) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([3]),np.array([1,0]))
    >>> f = op.log2(x)
    >>> f
    FuncInput([1.5849625], [0.48089835,0])
    """
    if isinstance(x, FuncInput):
        new_vals = np.log2(x.val_)
        new_ders = [x.ders_[i] * (1/(x.val_ * np.log(2))) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.log2(x)

def logn(x, base): # log with arbitrary base
    return log(x)/log(base)

@validate_input
def log1p(x):
    """
    Returns the log(1+x) of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on log(1+x) and gradients based on 1/(1+x) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([0]),np.array([1,0]))
    >>> f = op.log1p(x)
    >>> f
    FuncInput([0], [1,0])
    """
    return log(1 + x)

def logaddexp(x1, x2):
    """
    Returns the log(exp(x1) + exp(x2)) of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on log(exp(x1) + exp(x2))
    and gradients based on exp(x)/(exp(x)+exp(y)) with respect to x and exp(y)/(exp(x)+exp(y)) with respect to y

    Examples
    =======
    >>> x = FuncInput(np.array([0]),np.array([1,0]))
    >>> y = FuncInput(np.array([1]),np.array([0,1]))
    >>> f = op.logaddexp(x,y)
    >>> f
    FuncInput([1.31326169], [0.26894142,0.73105858])
    """
    return log(exp(x1) + exp(x2))

def logaddexp2(x1, x2):
    """
    Returns the log(x1**2 + x2**2) of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on log(x1**2 + x2**2)
    and gradients based on 2x/(x**2+y**2) with respect to x and 2y/(x**2+y**2) with respect to y

    Examples
    =======
    >>> x = FuncInput(np.array([2]),np.array([1,0]))
    >>> y = FuncInput(np.array([1]),np.array([0,1]))
    >>> f = op.logaddexp(x,y)
    >>> f
    FuncInput([2.32192809], [1.15415603,0.57707802])
    """
    return log2(x1**2 + x2**2)

def logistic(x): # standard logistic function
    return 1/(1 + exp(-x))

# Trigonometric functions
@validate_input
def sin(x):
    """
    Returns the sine of FuncInput object.

    Parameters
    =======
    FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on sin(x) and gradients based on cos(x) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([np.pi/6]),np.array([1,0]))
    >>> f = op.sin(x)
    >>> f
    FuncInput([0.5], [0.8660254,0.])
    """
    if isinstance(x, FuncInput):
        new_vals = np.sin(x.val_)
        new_ders = [x.ders_[i] * np.cos(x.val_) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.sin(x)

@validate_input
def cos(x):
    """
    Returns the cos of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on cos(x) and gradients based on -sin(x) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([np.pi/3]),np.array([1,0]))
    >>> f = op.cos(x)
    >>> f
    FuncInput([0.5], [-0.8660254,0.])
    """
    if isinstance(x, FuncInput):
        new_vals = np.cos(x.val_)
        new_ders = [x.ders_[i] * (-np.sin(x.val_)) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.cos(x)

def tan(x):
    """
    Returns the tan of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on tan(x) and gradients based on 1/(cos(x))^2 * x'

    Examples
    =======
    >>> x = FuncInput(np.array([np.pi/3]),np.array([1,0]))
    >>> f = op.tan(x)
    >>> f
    FuncInput([1.73205081], [4.,0.])
    """
    if isinstance(x, FuncInput):
        new_vals = np.tan(x.val_)
        new_ders = [x.ders_[i] * (1/np.cos(x.val_))**2 for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.tan(x)

def arcsin(x):
    """
    Returns the arcsin of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on arcsin(x) and gradients based on 1/sqrt(1-x^2) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([0.5]),np.array([1,0]))
    >>> f = op.arcsin(x)
    >>> f
    FuncInput([0.5235988], [0.8660254, 0.])
    """
    if isinstance(x, FuncInput):
        assert x.val_ > -1 and x.val_ < 1, 'Input is outside the domain of arcsin or its derivative'
        new_val = np.arcsin(x.val_)
        new_ders = [(1/sqrt(1 - x.val_**2)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x >= -1 and x <= 1, 'Input is outside the domain of arcsin'
        return np.arcsin(x)

def arccos(x):
    """
    Returns the arccos of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on arccos(x) and gradients based on -1/sqrt(1-x^2) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([0.5]),np.array([1,0]))
    >>> f = op.arcsin(x)
    >>> f
    FuncInput([1.04719755], [-0.8660254, 0.])
    """
    if isinstance(x, FuncInput):
        assert x.val_ > -1 and x.val_ < 1, 'Input is outside the domain of arccos or its derivative'
        new_val = np.arccos(x.val_)
        new_ders = [(-(1/sqrt(1 - x.val_**2))) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x >= -1 and x <= 1, 'Input is outside the domain of arccos'
        return np.arccos(x)

def arctan(x):
    """
    Returns the arctan of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on arctan(x) and gradients based on 1/(1+x^2) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([0.5]),np.array([1,0]))
    >>> f = op.arcsin(x)
    >>> f
    FuncInput([0.7853982], [0.5, 0.])
    """
    if isinstance(x, FuncInput):
        new_val = np.arctan(x.val_)
        new_ders = [(1/(1 + x.val_**2)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.arctan(x)

def hypot(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')

def arctan2(x1, x2):
    raise NotImplementedError('Function not yet implemented in differtless')

# Hyperbolic functions

def sinh(x):
    """
    Returns the sinh of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on sinh(x) = (exp(x) - exp(-x)) * 1/2 and gradients based on cosh(x) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([1]),np.array([1,0]))
    >>> f = op.sinh(x)
    >>> f
    FuncInput([1.17520119], [1.54308063, 0.])
    """
    if isinstance(x, FuncInput):
        new_val = np.sinh(x.val_)
        new_ders = [np.cosh(x.val_) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.sinh(x)

def cosh(x):
    """
    Returns the cosh of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on sinh(x) = (exp(x) + exp(-x)) * 1/2 and gradients based on sinh(x) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([1]),np.array([1,0]))
    >>> f = op.cosh(x)
    >>> f
    FuncInput([1.54308063], [1.17520119, 0.])
    """
    if isinstance(x, FuncInput):
        new_val = np.cosh(x.val_)
        new_ders = [(np.sinh(x.val_)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.cosh(x)

def tanh(x):
    """
    Returns the tanh of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on tanh(x) = sinh(x) / cosh(x) and gradients based on 1 / cosh(x) ** 2 * x'

    Examples
    =======
    >>> x = FuncInput(np.array([2]),np.array([1,0]))
    >>> f = op.tanh(x)
    >>> f
    FuncInput([0.9640276], [0.0706508, 0.])
    """
    if isinstance(x, FuncInput):
        new_val = np.tanh(x.val_)
        new_ders = [((1/np.cosh(x.val_)) ** 2) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.tanh(x)

def arcsinh(x):
    """
    Returns the arcsinh of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on arcsinh(x) = log(x + sqrt(x**2+1)) and gradients based on 1 / sqrt(x**2+1) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([2]),np.array([1,0]))
    >>> f = op.tanh(x)
    >>> f
    FuncInput([0.88137359], [0.70710678, 0.])
    """
    if isinstance(x, FuncInput):
        new_val = np.arcsinh(x.val_)
        new_ders = [(1/sqrt(x.val_**2 + 1)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.arcsinh(x)

def arccosh(x):
    """
    Returns the arccosh of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on arcsinh(x) = log(x + sqrt(x**2-1)) and gradients based on 1 / sqrt(x**2-1) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([2]),np.array([1,0]))
    >>> f = op.tanh(x)
    >>> f
    FuncInput([1.31695790], [0.57735027, 0.])
    """
    if isinstance(x, FuncInput):
        assert x.val_ > 1, 'Input is outside the domain of arccosh or its derivative'
        new_val = np.arccosh(x.val_)
        new_ders = [(1/sqrt((x.val_**2) - 1)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x >= 0, 'Input is outside the domain of arccosh'
        return np.arccosh(x)

def arctanh(x):
    """
    Returns the arctanh of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on arcsinh(x) = log((1+x)/(1-x))/2 and gradients based on 1 / (1-x**2) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([0.5]),np.array([1,0]))
    >>> f = op.tanh(x)
    >>> f
    FuncInput([0.549306], [1.33333, 0.])
    """
    if isinstance(x, FuncInput):
        assert np.abs(x.val_) < 1, 'Input is outside the domain of arctanh or its derivative'
        new_val = np.arctanh(x.val_)
        new_ders = [(1/(1-x.val_**2)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert abs(x) < 1, 'Input is outside the domain of arctanh or its derivative'
        return np.arctanh(x)

'''
Misc functions (not from numpy)
'''

def validate_input_multiple(func):
    '''
    Same as validate_input but with flexibility for additional function arguments
    (keeping these separate to increase robustness of validate_input to poorly specified inputs
    in single argument functions)
    '''
    def wrapper(x, *args):
        if not isinstance(x, FuncInput) and not isinstance(x, numbers.Real):
            raise TypeError('Inputs must be type FuncInput or a real number')
        return func(x, *args)
    return wrapper


@validate_input
def erf(x):
    """
    Returns the error function of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value based on error function from scipy and gradients based on 2 * exp(-x**2) / sqrt(pi) * x'

    Examples
    =======
    >>> x = FuncInput(np.array([1,20]),np.array([1]))
    >>> f = op.erf(x)
    >>> f
    FuncInput([0.84270079, 1.], [0.4151075, 0.])
    """
    if isinstance(x, FuncInput):
        new_vals = special.erf(x.val_)
        new_ders = [x.ders_[i] * 2/(np.pi**0.5) * np.exp(- (x.val_)**2) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return special.erf(x)


@validate_input
def gamma(x):
    """
    Returns the gamma function of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value and gradients based on gamma function from scipy

    Examples
    =======
    >>> x = FuncInput(np.array([1,4]),np.array([1]))
    >>> f = op.gamma(x)
    >>> f
    FuncInput([1., 6.], [0.63353918, 3.96259814])
    """
    if isinstance(x, FuncInput):
        new_vals = special.gamma(x.val_)
        new_ders = [x.ders_[i] * 2/(np.pi**0.5) * np.exp(special.digamma(x.val_)) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return special.gamma(x)


def factorial(x):
    """
    Returns the factorial function of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value and gradients from x! based on gamma function from scipy

    Examples
    =======
    >>> x = FuncInput(np.array([0,3]),np.array([1]))
    >>> f = op.factorial(x)
    >>> f
    Value:
    [1., 6.]
    Gradient(s):
    [0.63353918, 3.96259814]
    """
    return gamma(x+1)


@validate_input
def floor(x):
    """
    Returns the floor value of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value and gradients from x! based on gamma function from scipy

    Examples
    =======
    >>> x = FuncInput(np.array([0,3]),np.array([1]))
    >>> f = op.tanh(x)
    >>> f
    Value:
    [0. 3.]
    Gradient(s):
    0
    """
    if isinstance(x, FuncInput):
        new_vals = np.floor(x.val_)
        warnings.warn('Using zero as derivatives for floor function (technically not defined at non-integers)...')
        new_ders = [x.ders_[i] * 0 for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.floor(x)


@validate_input_multiple
def gammainc(x, alpha): # lower incomplete gamma function
    """
    Returns the lower incomplete gamma function of FuncInput object.

    Parameters
    =======
    x:
        FuncInput object or real number

    Returns
    =======
    FuncInput object with value and gradients based on gammainc function from scipy

    Examples
    =======
    >>> f = op.gammainc(3,2)
    >>> f
    0.8008517265285442
    """
    if isinstance(x, FuncInput):
        new_vals = special.gammainc(alpha, x.val_)
        new_ders = [x.ders_[i] * (x**(alpha-1))*exp(-x) for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return special.gammainc(alpha, x)


'''
Statistical distributions
'''

class Normal():
    """ 
    Returns the features of normal distribution function of FuncInput object.
    
    Parameters
    =======
    x: FuncInput object or real number
        
    METHODS
    ========
    pdf: probability density function
    logpdf: log of probability density function
    cdf: cumulative density function
    logcdf: log of cumulative density function
    
    Returns
    =======
    FuncInput object with value and gradients based on the selected method
    
    Examples
    =======
    >>> x = FuncInput(np.array([1,20]),np.array([1]))
    >>> f = op.Normal().pdf(x)
    Value:
    [0.24197072 0.        ]
    Gradient(s):
    [-0.24197072 -0.        ]
    """

    def __init__(self, loc=0, scale=1):
        '''Normal distribution with mean `loc` and standard deviation `scale`'''
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
    """ 
    Returns the features of Poisson distribution function of FuncInput object.
    
    Parameters
    =======
    x: FuncInput object or real number
        
    METHODS
    ========
    pmf: probability mass function
    logpmf: log of probability mass function
    cdf: cumulative density function
    logcdf: log of cumulative density function
    
    Returns
    =======
    FuncInput object with value and gradients based on the selected method
    
    Examples
    =======
    >>> x = FuncInput(np.array([1,20]),np.array([1]))
    >>> f = op.Poisson(mu=2).pmf(x)
    Value:
    [0.27067057 0.        ]
    Gradient(s):
    [-0.27851754  0.        ]
    """
    def __init__(self, mu):
        '''Poisson distribution with shape parameter `mu`'''
        self.mu = mu

    def __str__(self):
        return f'Poisson distribution with shape parameter {self.mu}'

    def __repr__(self):
        return f'Poisson(mu={self.mu})'

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


class Gamma():
    """ 
    Returns the features of Poisson distribution function of FuncInput object.
    
    Parameters
    =======
    x: FuncInput object or real number
        
    METHODS
    ========
    pdf: probability density function
    logpdf: log of probability density function
    cdf: cumulative density function
    logcdf: log of cumulative density function
    
    Returns
    =======
    FuncInput object with value and gradients based on the selected method
    
    Examples
    =======
    >>> x = FuncInput(np.array([1,20]),np.array([1]))
    >>> f = op.Gamma(alpha=1, beta=1).pdf(x)
    Value:
    [0.36787944 0.        ]
    Gradient(s):
    [-0.36787944 -0.        ]
    """
    def __init__(self, alpha=0, beta=1):
        '''Gamma distribution with shape `alpha` and scale `beta`'''
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return f'Gamma distribution with shape {self.alpha} and scale {self.beta}'

    def __repr__(self):
        return f'Gamma(shape={self.alpha}, scale={self.beta})'

    def pdf(self, x):
        return (1/(gamma(self.alpha)*self.beta))*((x/self.beta)**(self.alpha-1))*exp(-x/self.beta)

    def logpdf(self, x):
        return -log(gamma(self.alpha)*self.beta) + (self.alpha-1)*log(x/self.beta) + (-x/self.beta)

    def cdf(self, x):
        return (1/gamma(self.alpha))*gammainc(x/self.beta,self.alpha)

    def logcdf(self, x):
        return -log(gamma(self.alpha)) + log(gammainc(x/self.beta,self.alpha))

'''
Distance Functions
'''

def euclidean(x, y):
    """
    Returns the Euclidean distance between any two real number or FuncInput
    objects.

    Parameters
    =======
    FuncInput object and/or real number. Parameters need not be the same dimension.

    Returns
    =======
    Euclidean distance between two points as well as the derivative.

    Examples
    =======
    >>> x = FuncInput(np.array([1,2,3]),np.array([1,0]))
    >>> y = FuncInput(np.array([9,8,7]),np.array([0,1])
    >>> f = op.euclidean(x,y)
    >>> f
    FuncInput([10.7703], [-1.6713, 1.6713])
    """
    def match_lengths(x, y):
        len_diff = len(x) - len(y)
        pad = [0] * abs(len_diff)
        x_val = np.append(x, pad) if len_diff < 0 else x
        y_val = np.append(y, pad) if len_diff > 0 else y
        new_der_num = sum(x_val - y_val)

        return x_val, y_val, new_der_num

    x_func = isinstance(x, FuncInput)
    y_func = isinstance(y, FuncInput)
    if not x_func and not y_func:
        try:
            iter(y)
            y_val = np.array(y)
        except:
            y_val = np.array([y])
            try:
                iter(x)
                x_val = np.array(x)
            except:
                x_val = np.array([x])
        x_val, y_val, new_der_num = match_lengths(x_val, y_val)
        return distance.euclidean(x_val, y_val)
    elif x_func and not y_func:
        try:
            iter(y)
            y_val = np.array(y)
        except TypeError:
            y_val = np.array([y])

        x_val, y_val, new_der_num = match_lengths(x.value, y_val)
        new_val = distance.euclidean(x_val, y_val)
        new_ders = [(new_der_num/new_val) * der for der in x.ders_]
    elif y_func and not x_func:
        try:
            iter(x)
            x_val = np.array(x)
        except TypeError:
            x_val = np.array([x])

        x_val, y_val, new_der_num = match_lengths(x_val, y.value)
        new_val = distance.euclidean(x_val, y_val)
        new_ders = [(new_der_num/new_val)  * (-der) for der in y.ders_]
    else:
        x_val, y_val, new_der_num = match_lengths(x.val_, y.val_)
        new_val = distance.euclidean(x_val, y_val)
        new_ders = [(new_der_num/new_val) * (x.gradients[i] - y.gradients[i]) for i in range(len(x.ders_))]

    return FuncInput(new_val, new_ders)
