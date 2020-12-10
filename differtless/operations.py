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
        new_ders = [(1/sqrt(1 - x.val_**2)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x >= -1 and x <= 1, 'Input is outside the domain of arcsin'
        return np.arcsin(x)

def arccos(x):
    if isinstance(x, FuncInput):
        assert x.val_ > -1 and x.val_ < 1, 'Input is outside the domain of arccos or its derivative'
        new_val = np.arccos(x.val_)
        new_ders = [(-(1/sqrt(1 - x.val_**2))) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x >= -1 and x <= 1, 'Input is outside the domain of arccos'
        return np.arccos(x)

def arctan(x):
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
    if isinstance(x, FuncInput):
        new_val = np.sinh(x.val_)
        new_ders = [np.cosh(x.val_) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.sinh(x)

def cosh(x):
    if isinstance(x, FuncInput):
        new_val = np.cosh(x.val_)
        new_ders = [(np.sinh(x.val_)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.cosh(x)

def tanh(x):
    if isinstance(x, FuncInput):
        new_val = np.tanh(x.val_)
        new_ders = [((1/np.cosh(x.val_)) ** 2) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.tanh(x)

def arcsinh(x):
    if isinstance(x, FuncInput):
        new_val = np.arcsinh(x.val_)
        new_ders = [(1/sqrt(x.val_**2 + 1)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        return np.arcsinh(x)

def arccosh(x):
    if isinstance(x, FuncInput):
        assert x.val_ > 1, 'Input is outside the domain of arccosh or its derivative'
        new_val = np.arccosh(x.val_)
        new_ders = [(1/sqrt((x.val_**2) - 1)) * x_der for x_der in x.ders_]
        return FuncInput(new_val, new_ders)
    elif isinstance(x, numbers.Real):
        assert x >= 0, 'Input is outside the domain of arccosh'
        return np.arccosh(x)

def arctanh(x):
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
        warnings.warn('Using zero as derivatives for floor function (technically not defined at non-integers)...')
        new_ders = [x.ders_[i] * 0 for i in range(len(x.ders_))]
        return FuncInput(new_vals, new_ders)
    elif isinstance(x, numbers.Real):
        return np.floor(x)


@validate_input_multiple
def gammainc(x, alpha): # lower incomplete gamma function
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

    def __init__(self, mu):
        '''Poisson distribution with shape parameter `mu`'''
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


class Gamma():

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
    def match_lengths(x, y):
        len_diff = len(x) - len(y)
        pad = [0] * abs(len_diff)
        x_val = np.append(x, pad) if len_diff < 0 else x
        y_val = np.append(y, pad) if len_diff > 0 else y

        return x_val, y_val

    x_func = isinstance(x, FuncInput)
    y_func = isinstance(y, FuncInput)
    if not x_func and not y_func:
        x_val, y_val = match_lengths(x, y)
        return distance.euclidean(x_val, y_val)
    elif x_func and not y_func:
        try:
            iter(y)
            y_val = np.array(y)
        except TypeError:
            y_val = np.array([y])

        x_val, y_val = match_lengths(x.value, y_val)
        new_val = distance.euclidean(x.value, y)
        new_ders = [2 * (x_val - y_val) * der for der in x.ders_]
    elif y_func and not x_func:
        try:
            iter(x)
            x_val = np.array(x)
        except TypeError:
            x_val = np.array([x])

        x_val, y_val = match_lengths(x_val, y.value)
        new_val = distance.euclidean(x, y.value)
        new_ders = [2 * (x_val - y_val) * (-der) for der in y.ders_]
    else:
        x_val, y_val = match_lengths(x.value, y.value)
        new_val = distance.euclidean(x_val, y_val)
        new_ders = [2 * (x_val - y_val) * (x.gradients[i] - y.gradients[i]) for i in range(len(x.ders_))]

    return FuncInput(new_val, new_ders)
