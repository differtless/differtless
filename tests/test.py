import numpy as np
import pytest
import math
from scipy.spatial import distance
from scipy import stats
from scipy.misc import derivative
import sys
sys.path.append('../')
import warnings
from differtless import ad
from differtless.ad import FuncInput, preprocess, forward, Jacobian, minimize
import differtless.operations as op

def test_add():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x + y
    assert f.value == [3], "add function is not correct"
    assert (f.gradients == np.array([1,1])).all(), "add function is not correct"

def test_sub():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x - y
    assert f.value == [-1], "sub function is not correct"
    assert (f.gradients == np.array([1,-1])).all(), "sub function is not correct"

def test_mul():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x * y
    assert f.value == 2, "mul function is not correct"
    assert (f.gradients == np.array([2,1])).all(), "mul function is not correct"

def test_truediv():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f1 = x / y
    f2 = x / 2
    assert f1.value == [0.5], "truediv function is not correct"
    assert (f1.gradients == np.array([0.5,-0.25])).all(), "truediv function is not correct"
    assert f2.value == [0.5], "truediv function is not correct"
    assert (f2.gradients == np.array([0.5,0])).all(), "truediv function is not correct"

def test_floordiv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    y = FuncInput(np.array([-5]),np.array([0,1]))
    f = y // x
    # f2 = y // 2
    assert f.value == [-3], "floordiv function is not correct"
    assert (f.gradients == np.array([1,0])).all(), "floordiv function is not correct"
    # assert f2.value == -3, "floordiv function is not correct"
    # assert (f2.gradients == np.array([0,0.5])).all(), "floordiv function is not correct"

def test_pow():
    x = FuncInput(np.array([2]),np.array([1,0]))
    y = FuncInput(np.array([3]),np.array([0,1]))
    f = x ** 3
    f_2 = x ** y
    assert f.value == [8], "pow function is not correct"
    assert (f.gradients == np.array([12,0])).all(), "pow function is not correct"
    assert f_2.value == [8], "pow function is not correct"

def test_neg():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = -x
    assert f.value == [-2], "neg function is not correct"
    assert (f.gradients == np.array([-1,0])).all(), "neg function is not correct"

def test_pos():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = + x
    assert f.value == [2], "pos function is not correct"
    assert (f.gradients == np.array([1,0])).all(), "pos function is not correct"

def test_abs():
    x = FuncInput(np.array([-2]),np.array([1,0]))
    f = abs(x)
    y = FuncInput(np.array([2]),np.array([1,0]))
    f2 = abs(y)
    assert f.value == [2], "abs function is not correct"
    assert (f.gradients == np.array([-1,0])).all(), "abs function is not correct"
    assert f2.value == [2], "abs function is not correct"
    assert (f2.gradients == np.array([1,0])).all(), "abs function is not correct"

def test_radd():
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = 1 + y
    assert f.value == [3], "radd function is not correct"
    assert (f.gradients == np.array([0,1])).all(), "radd function is not correct"

def test_rsub():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = 1 - y
    f2 = x - y
    assert f.value == [-1], "rsub function is not correct"
    assert (f.gradients == np.array([0,-1])).all(), "rsub function is not correct"
    assert f2.value == [-1], "rsub function is not correct"
    assert (f2.gradients == np.array([1,-1])).all(), "rsub function is not correct"

def test_rmul():
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = 2 * y
    assert f.value == [4], "rmul function is not correct"
    assert (f.gradients == np.array([0,2])).all(), "rmul function is not correct"

def test_rtruediv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = 1 / x
    assert f.value == [0.5], "rtruediv function is not correct"
    assert (f.gradients == np.array([-0.25,0])).all(), "rtruediv function is not correct"

def test_rfloordiv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = -5 // x
    assert f.value == [-3], "rfloordiv function is not correct"
    assert (f.gradients == np.array([1,0])).all(), "rfloordiv function is not correct"

def rpow():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = 2**x
    assert f.value == [4], "rpow function is not correct"
    assert (abs(f.gradients - np.array([2.77258872,0]))<1e-6).all(), "rpow function is not correct"

def test_exp():
    x = FuncInput(np.array([0]),np.array([1,0]))
    f = op.exp(x)
    f2 = op.exp(1)
    assert f.value == [1], "exp function is not correct"
    assert (f.gradients == np.array([1,0])).all(), "exp function is not correct"
    assert f2 == np.exp(1), "exp function is not correct"

def test_expm1():
    x = FuncInput(np.array([0]),np.array([1,0]))
    f = op.expm1(x)
    assert f.value == [0], "expm1 function is not correct"
    assert (f.gradients == np.array([1,0])).all(), "expm1 function is not correct"

def test_exp2():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = op.exp2(x)
    assert f.value == [4], "exp2 function is not correct"
    assert (abs(f.gradients - np.array([2.77258872,0]))<1e-6).all(), "exp2 function is not correct"

def test_log():
    x = FuncInput(np.array([3]),np.array([1,0]))
    f = op.log(x)
    f2 = op.log(3)
    assert (abs(f.value - 1.09861229) < 1e-6).all(), "log function is not correct"
    assert (abs(f.gradients - np.array([0.33333333,0]))<1e-6).all(), "log function is not correct"
    assert (abs(f2-np.log(3))<1e-6), "log function is not correct"

def test_log10():
    x = FuncInput(np.array([3]),np.array([1,0]))
    f = op.log10(x)
    f2 = op.log10(3)
    assert (abs(f.value - 0.47712125) < 1e-6).all(), "log10 function is not correct"
    assert (abs(f.gradients - np.array([0.14476483,0]))<1e-6).all(), "log10 function is not correct"
    assert (abs(f2-np.log10(3))<1e-6), "log10 function is not correct"

def test_log2():
    x = FuncInput(np.array([3]),np.array([1,0]))
    f = op.log2(x)
    f2 = op.log2(3)
    assert (abs(f.value - 1.5849625) < 1e-6).all(), "log2 function is not correct"
    assert (abs(f.gradients - np.array([0.48089835,0]))<1e-6).all(), "log2 function is not correct"
    assert (abs(f2-np.log2(3))<1e-6), "log2 function is not correct"

def test_log1p():
    x = FuncInput(np.array([0]),np.array([1,0]))
    f = op.log1p(x)
    assert f.value == [0], "log1p function is not correct"
    assert (f.gradients == np.array([1,0])).all(), "log1p function is not correct"

def test_logaddexp():
    x = FuncInput(np.array([0]),np.array([1,0]))
    y = FuncInput(np.array([1]),np.array([0,1]))
    f = op.logaddexp(x,y)
    assert (abs(f.value - 1.31326169) < 1e-6).all(), "logaddexp function is not correct"
    assert (abs(f.gradients - np.array([0.26894142,0.73105858]))<1e-6).all(), "logaddexp function is not correct"

def test_logaddexp2():
    x = FuncInput(np.array([2]),np.array([1,0]))
    y = FuncInput(np.array([1]),np.array([0,1]))
    f = op.logaddexp2(x,y)
    assert (abs(f.value - 2.32192809) < 1e-6).all(), "logaddexp2 function is not correct"
    assert (abs(f.gradients - np.array([1.15415603,0.57707802]))<1e-6).all(), "logaddexp2 function is not correct"

def test_logn():
    x = FuncInput(3,[1])
    assert abs(op.logn(x,3).value - 1.0)<1e-6
    assert abs(op.logn(x,3).gradients - 0.3034130755422791)<1e-6
    assert abs(op.logn(3,3) - 1.0)<1e-6

def test_logistic():
    x = FuncInput(0,[1])
    assert abs(op.logistic(x).value - 0.5)<1e-6
    assert abs(op.logistic(x).gradients - 0.25)<1e-6
    assert abs(op.logistic(0) - 0.5)<1e-6

def test_expn():
    x = FuncInput(3,[1])
    assert abs(op.expn(x,3).value - 27)<1e-6
    assert abs(op.expn(x,3).gradients - 29.662531794038966)<1e-6
    assert abs(op.expn(3,3) - 27)<1e-6

def test_sin():
    x = FuncInput(np.array([np.pi/6]),np.array([1,0]))
    f = op.sin(x)
    f2 = op.sin(np.pi/6)
    assert (abs(f.value - 0.5)<1e-6).all(), "sin function is not correct"
    assert (abs(f.gradients - np.array([0.8660254,0.]))<1e-6).all(), "sin function is not correct"
    assert (abs(f2-np.sin(np.pi/6))<1e-6), "sin function is not correct"

def test_cos():
    x = FuncInput(np.array([np.pi/3]),np.array([1,0]))
    f = op.cos(x)
    f2 = op.cos(np.pi/3)
    assert (abs(f.value - 0.5)<1e-6).all(), "cos function is not correct"
    assert (abs(f.gradients - np.array([-0.8660254,0.]))<1e-6).all(), "cos function is not correct"
    assert (abs(f2-np.cos(np.pi/3))<1e-6), "sin function is not correct"

def test_tan():
    x = FuncInput(np.array([np.pi/3]),np.array([1,0]))
    f = op.tan(x)
    f2 = op.tan(np.pi/3)
    assert (abs(f.value - 1.73205081) < 1e-6).all(), "tan function is not correct"
    assert (abs(f.gradients - np.array([4.,0.]))<1e-6).all(), "tan function is not correct"
    assert (abs(f2-np.tan(np.pi/3))<1e-6), "sin function is not correct"

def test_arcsin():
    x = FuncInput(np.array([0.5]), np.array([1,0]))
    f = op.arcsin(x)
    assert (abs(f.value - (np.pi/6)) < 1e-6).all(), 'arcsin function is not correct'
    assert (abs(f.gradients - np.array([1/math.sqrt(1 - 0.5**2), 0])) < 1e-6).all(), 'arcsin function not correct'
    with pytest.raises(AssertionError):
        x = FuncInput(np.array([1]),np.array([1,0]))
        f = op.arcsin(x)
    xreal = 0.5
    freal = op.arcsin(xreal)
    assert (abs(freal - (np.pi/6)) < 1e-6).all(), 'arcsin function is not correct'

def test_arccos():
    x = FuncInput(np.array([0.5]), np.array([1,0]))
    f = op.arccos(x)
    assert (abs(f.value - (np.pi/3)) < 1e-6).all(), 'arccos function is not correct'
    assert (abs(f.gradients - np.array([-(1/math.sqrt(1 - 0.5**2)), 0])) < 1e-6).all(), 'arccos function not correct'
    with pytest.raises(AssertionError):
        x = FuncInput(np.array([1]),np.array([1,0]))
        f = op.arccos(x)
    xreal = 0.5
    freal = op.arccos(xreal)
    assert (abs(freal - (np.pi/3)) < 1e-6).all(), 'arcsin function is not correct'

def test_arctan():
    x = FuncInput(np.array([1]),np.array([1,0]))
    f = op.arctan(x)
    assert (abs(f.value - np.pi/4) < 1e-6) , 'arctan function is not correcct'
    assert (abs(f.gradients - np.array([0.5, 0])) < 1e-6).all(), 'arctan function is not correct'
    xreal = 1
    freal = op.arctan(xreal)
    assert (abs(freal - (np.pi/4)) < 1e-6).all(), 'arcsin function is not correct'

def test_hypot():
    with pytest.raises(NotImplementedError):
        x = FuncInput(np.array([1]),np.array([1]))
        x2 = FuncInput(np.array([2]),np.array([1]))
        f = op.hypot(x, x2)

def test_arctan2():
    with pytest.raises(NotImplementedError):
        x = FuncInput(np.array([1]),np.array([1,0]))
        x2 = FuncInput(np.array([2]),np.array([1]))
        f = op.arctan2(x, x2)

# Hyperbolic functions

def test_sinh():
    x = FuncInput(np.array([1]),np.array([1,0]))
    f = op.sinh(x)
    assert (abs(f.value - (-1 + np.exp(2))/(2*np.exp(1))) < 1e-6).all(), 'sinh function is not correct'
    assert (abs(f.gradients - np.array([op.cosh(x).value, 0])) < 1e-6).all(), 'sinh function is not correct'
    xreal = 1
    freal = op.sinh(xreal)
    assert (abs(freal - (-1 + np.exp(2))/(2*np.exp(1))) < 1e-6).all(), 'sinh function is not correct'

def test_cosh():
    x = FuncInput(np.array([1]),np.array([1,0]))
    f = op.cosh(x)
    assert (abs(f.value - (1 + np.exp(2))/(2*np.exp(1))) < 1e-6).all(), 'cosh function is not correct'
    assert (abs(f.gradients - np.array([op.sinh(x).value, 0])) < 1e-6).all(), 'cosh function is not correct'
    xreal = 1
    freal = op.cosh(xreal)
    assert (abs(freal - (1 + np.exp(2))/(2*np.exp(1))) < 1e-6).all(), 'cosh function is not correct'

def test_tanh():
    x = FuncInput(np.array([1]),np.array([1,0]))
    f = op.tanh(x)
    assert (abs(f.value - np.tanh(1)) < 1e-6).all(), 'tanh function is not correct'
    assert (abs(f.gradients - np.array([(1/np.cosh(1)) **2, 0])) < 1e-6).all(), 'tanh function is not correct'
    xreal = 1
    freal = op.tanh(xreal)
    assert (abs(freal - np.tanh(1)) < 1e-6).all(), 'tanh function is not correct'

def test_arcsinh():
    x = FuncInput(np.array([1]),np.array([1,0]))
    f = op.arcsinh(x)
    assert (abs(f.value - (np.arcsinh(1))) < 1e-6).all(), 'arcsinh function is not correct'
    assert (abs(f.gradients - np.array([(1/math.sqrt(2)), 0])) < 1e-6).all(), 'arcsinh function is not correct'
    xreal = 1
    freal = op.arcsinh(xreal)
    assert (abs(freal - (np.arcsinh(1))) < 1e-6).all(), 'arcsinh function is not correct'

def test_arccosh():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = op.arccosh(x)
    assert (abs(f.value - np.arccosh(2)) < 1e-6).all(), 'arccosh function is not correct'
    assert (abs(f.gradients - np.array([1/math.sqrt(3), 0])) < 1e-6).all(), 'arcosh function is not correct'
    with pytest.raises(AssertionError):
        x = FuncInput(np.array([1]),np.array([1,0]))
        f = op.arccosh(x)
    xreal = 2
    freal = op.arccosh(xreal)
    assert (abs(freal - np.arccosh(2)) < 1e-6).all(), 'arccosh function is not correct'

def test_arctanh():
    x = FuncInput(np.array([0.5]),np.array([1,0]))
    f = op.arctanh(x)
    assert (abs(f.value - np.arctanh(0.5)) < 1e-6).all(), 'arctanh function is not correct'
    assert (abs(f.gradients - np.array([1/(1-0.5**2), 0])) < 1e-6).all(), 'arctanh function is not correct'
    with pytest.raises(AssertionError):
        x = FuncInput(np.array([1]),np.array([1,0]))
        f = op.arctanh(x)
    xreal = 0.5
    freal = op.arctanh(xreal)
    assert (abs(freal - np.arctanh(0.5)) < 1e-6).all(), 'arctanh function is not correct'

# Misc functions

def test_erf():
    x = FuncInput(np.array([1,20]),np.array([1]))
    f = op.erf(x)
    assert (op.erf(0) == 0), "erf function is not correct"
    assert (abs(f.value - np.array([0.84270079, 1.]))<1e-6).all(), "erf function is not correct"
    assert (abs(f.gradients - np.array([0.4151075, 0.]))<1e-6).all(), "erf function is not correct"

def test_gamma():
    x = FuncInput(np.array([1,4]),np.array([1]))
    f = op.gamma(x)
    assert (op.gamma(4.0) == 6.0), "gamma function is not correct"
    assert (abs(f.value - np.array([1., 6.]))<1e-6).all(), "gamma function is not correct"
    assert (abs(f.gradients - np.array([0.63353918, 3.96259814]))<1e-6).all(), "gamma function is not correct"

def test_factorial():
    x = FuncInput(np.array([0,3]),np.array([1]))
    f = op.factorial(x)
    assert (op.factorial(3.0) == 6.0), "factorial function is not correct"
    assert (abs(f.value - np.array([1., 6.]))<1e-6).all(), "factorial function is not correct"
    assert (abs(f.gradients - np.array([0.63353918, 3.96259814]))<1e-6).all(), "factorial function is not correct"

def test_floor():
    x = FuncInput(np.array([0,3]),np.array([1]))
    f = op.floor(x)
    assert (op.floor(2.2) == 2.0), "floor function is not correct"
    assert (abs(f.value - np.array([0., 3.]))<1e-6).all(), "floor function is not correct"
    with warnings.catch_warnings(record=True) as w:
        op.floor(x)
        assert len(w) > 0, "floor function does not display warning"

# AD functionality

def test_preprocess():
    inputs_1 = [1, 2]
    seed_1 = [[1,1],[2,2]]
    inputs_2 = [[1],(2)]
    assert preprocess(inputs_1)[0].value == np.array([1]), 'preprocess is mishandling seed = []'
    assert preprocess(inputs_1)[1].value == np.array([2]), 'preprocess is mishandling seed = []'
    assert (preprocess(inputs_1)[0].gradients == np.array([1,0])).all(), 'preprocess is mishandling seed = []'
    assert (preprocess(inputs_1)[1].gradients == np.array([0,1])).all(), 'preprocess is mishandling seed = []'
    assert preprocess(inputs_2)[0].value == np.array([1]), 'preprocess is mishandling seed = []'
    assert preprocess(inputs_2)[1].value == np.array([2]), 'preprocess is mishandling seed = []'

    assert preprocess(inputs_1, seed_1)[0].value == np.array([1]), 'preprocess is not creating correct gradients'
    assert preprocess(inputs_1, seed_1)[1].value == np.array([2]), 'preprocess is not creating correct gradients'
    assert (preprocess(inputs_1, seed_1)[0].gradients == np.array([1,1])).all(), 'preprocess is not creating correct gradients'
    assert (preprocess(inputs_1, seed_1)[1].gradients == np.array([2,2])).all(), 'preprocess is not creating correct gradients'

def test_preprocess_string_input():
    with pytest.raises(TypeError):
        inputs_1 = [1, '2']
        seed_1 = [[1,1],[2,2]]
        preprocess(inputs_1)

def test_preprocess_bad_seed():
    with pytest.raises(ValueError):
        inputs_1 = [1, 2]
        seed_1 = [[1,1]]
        preprocess(inputs_1, seed_1)

def test_preprocess_bad_seed2():
    with pytest.raises(ValueError):
        inputs_1 = [1, 2]
        seed_1 = [[1], [1, 2]]
        preprocess(inputs_1, seed_1)

def test_preprocess_bad_seed3():
    with pytest.raises(TypeError):
        inputs_1 = [1, 2]
        seed_1 = [[1,1], [2, '2']]
        preprocess(inputs_1, seed_1)

def test_forward():
    inputs = [1, 2]
    seeds = [[1, 0], [0, 1]]
    def simple_func(x, y):
        return (x + y) ** 2
    def simple_func2(x, y):
        return x + y
    assert forward(simple_func, inputs, seeds).value == np.array([9]), 'forward mode is not correct'
    assert (forward(simple_func, inputs, seeds).gradients == np.array([6.,6.])).all(), 'forward mode is not correct'
    assert (forward([simple_func,simple_func2], inputs, seeds).value == np.array([9,3])).all(), 'forward mode is not correct'
    assert (forward([simple_func,simple_func2], inputs, seeds).gradients == np.array([[6.,6.],[1.,1.]])).all(), 'forward mode is not correct'


# Optimization routines

def test_minimize():
    assert abs(ad.minimize(lambda x: (x-3)**2, 2)[0] - 3)<1e-6,'minimize function is not correct'
    assert abs(ad.minimize(lambda x: (x-3)**2, 2, descriptive=True)['x'][0] - 3)<1e-6,'minimize function is not correct'

def test_root():
    assert abs(ad.root(lambda x: (x-3)**2, 2)[0] - 3)<1e-6,'root function is not correct'
    assert abs(ad.root(lambda x: (x-3)**2, 2, descriptive=True)['x'][0] - 3)<1e-6,'root function is not correct'
    with pytest.raises(NotImplementedError):
        ad.root(lambda x: (x-3)**2, [2, 3, 4])

def test_least_squares():
    assert abs(ad.least_squares(lambda x: (x-3)**2, 2, bounds=[1,2.4])[0] - 2.4)<1e-6,'least squares function is not correct'
    abs(ad.least_squares(lambda x: (x-3)**2, 2, bounds=[1,2.4], descriptive=True)['x'][0] - 2.4)<1e-6,'least squares function is not correct'

# Probability distributions – placeholders to make sure functions are executable both for scalar and 'FuncInput'

def test_Normal():
    assert str(op.Normal())
    assert repr(op.Normal())
    x = FuncInput(np.array([1,20]),np.array([1]))
    assert op.Normal(loc=2, scale=4).pdf(4) == stats.norm(loc=2, scale=4).pdf(4), 'normal distribution pdf is not correct'
    assert (op.Normal().pdf(x).value == stats.norm().pdf([1,20])).all(), 'normal distribution pdf is not correct'
    assert (abs(op.Normal().pdf(x).gradients - [derivative(stats.norm.pdf,1,dx=1e-6),derivative(stats.norm.pdf,20,dx=1e-6)])<1e-6).all(), 'normal distribution pdf is not correct'
    assert op.Normal().logpdf(4) == stats.norm().logpdf(4), 'normal distribution logpdf is not correct'
    assert (op.Normal().logpdf(x).value == stats.norm().logpdf([1,20])).all(), 'normal distribution logpdf is not correct'
    assert (abs(op.Normal().logpdf(x).gradients - [derivative(stats.norm.logpdf,1,dx=1e-6),derivative(stats.norm.logpdf,20,dx=1e-6)])<1e-6).all(), 'normal distribution logpdf is not correct'
    assert op.Normal().cdf(4) == stats.norm().cdf(4), 'normal distribution cdf is not correct'
    assert (op.Normal().cdf(x).value == stats.norm().cdf([1,20])).all(), 'normal distribution cdf is not correct'
    assert (abs(op.Normal().cdf(x).gradients - [derivative(stats.norm.cdf,1,dx=1e-6),derivative(stats.norm.cdf,20,dx=1e-6)])<1e-6).all(), 'normal distribution cdf is not correct'
    assert (op.Normal().logcdf(4) - stats.norm().logcdf(4))<1e-6, 'normal distribution logcdf is not correct'
    assert (abs(op.Normal().logcdf(x).value - stats.norm().logcdf([1,20]))<1e-6).all(), 'normal distribution logcdf is not correct'
    assert (abs(op.Normal().logcdf(x).gradients - [derivative(stats.norm.logcdf,1,dx=1e-6),derivative(stats.norm.logcdf,20,dx=1e-6)])<1e-6).all(), 'normal distribution logcdf is not correct'

# x = FuncInput(np.array([1,20]),np.array([1]))
# print(op.Gamma(alpha=1, beta=1).pdf(x))

def test_Gamma():
    assert str(op.Gamma())
    assert repr(op.Gamma())
    x = FuncInput(np.array([1,20]),np.array([1]))
    assert op.Gamma(alpha=1, beta=1).pdf(4) == stats.gamma(1).pdf(4), 'gamma distribution pdf is not correct'
    assert (op.Gamma(alpha=1, beta=1).pdf(x).value == stats.gamma(1).pdf([1,20])).all(), 'gamma distribution pdf is not correct'
    assert (abs(op.Gamma(alpha=1, beta=1).pdf(x).gradients - [derivative(stats.gamma(1).pdf,1,dx=1e-6),derivative(stats.gamma(1).pdf,20,dx=1e-6)])<1e-6).all(), 'gamma distribution pdf is not correct'

    assert op.Gamma(alpha=1, beta=1).logpdf(4) == stats.gamma(1).logpdf(4), 'gamma distribution logpdf is not correct'
    assert (op.Gamma(alpha=1, beta=1).logpdf(x).value == stats.gamma(1).logpdf([1,20])).all(), 'gamma distribution logpdf is not correct'
    assert (abs(op.Gamma(alpha=1, beta=1).logpdf(x).gradients - [derivative(stats.gamma(1).logpdf,1,dx=1e-6),derivative(stats.gamma(1).logpdf,20,dx=1e-6)])<1e-6).all(), 'gamma distribution logpdf is not correct'

    assert op.Gamma(alpha=1, beta=1).cdf(4) == stats.gamma(1).cdf(4), 'gamma distribution cdf is not correct'
    assert (op.Gamma(alpha=1, beta=1).cdf(x).value == stats.gamma(1).cdf([1,20])).all(), 'gamma distribution cdf is not correct'
    # assert (abs(op.Gamma(alpha=1, beta=1).cdf(x).gradients[0] - [derivative(stats.gamma(1).cdf,1,dx=1e-6),derivative(stats.gamma(1).cdf,20,dx=1e-6)])<1e-6).all(), 'gamma distribution cdf is not correct'

    # assert op.Gamma(alpha=1, beta=1).logcdf(4) == stats.gamma(1).logcdf(4), 'gamma distribution logcdf is not correct'
    # assert (op.Gamma(alpha=1, beta=1).logcdf(x).value == stats.gamma(1).logcdf([1,20])).all(), 'gamma distribution logcdf is not correct'
    # assert (abs(op.Gamma(alpha=1, beta=1).logcdf(x).gradients[0] - [derivative(stats.gamma(1).logcdf,1,dx=1e-6),derivative(stats.gamma(1).logcdf,20,dx=1e-6)])<1e-6).all(), 'gamma distribution logcdf is not correct'



def test_Poisson():
    assert str(op.Poisson(mu=2))
    assert repr(op.Poisson(mu=2))
    x = FuncInput(np.array([1,20]),np.array([1]))
    # assert abs(op.Poisson(mu=2).pmf(4) - stats.poisson(2).pmf(4))<1e-6, 'poisson distribution pmf is not correct'
    # assert (op.Poisson(mu=2).pmf(x).value == stats.poisson(2).pdf([1,20])).all(), 'poisson distribution pmf is not correct'
    # assert (abs(op.Poisson(mu=2).pmf(x).gradients - [derivative(stats.poisson(mu=2).pmf,1,dx=1e-6),derivative(stats.poisson(mu=2).pmf,20,dx=1e-6)])<1e-6).all(), 'poisson distribution pmf is not correct'

    # assert abs(op.Poisson(mu=2).logpmf(4) - stats.poisson(2).logpmf(4))<1e-6, 'poisson distribution logpmf is not correct'
    # assert (op.Poisson(mu=2).logpmf(x).value == stats.poisson(2).pdf([1,20])).all(), 'poisson distribution logpmf is not correct'
    # assert (abs(op.Poisson(mu=2).logpmf(x).gradients - [derivative(stats.poisson(mu=2).logpmf,1,dx=1e-6),derivative(stats.poisson(mu=2).logpmf,20,dx=1e-6)])<1e-6).all(), 'poisson distribution logpmf is not correct'

    # assert abs(op.Poisson(mu=2).cdf(4) - stats.poisson(2).cdf(4))<1e-6, 'poisson distribution cdf is not correct'
    # assert (op.Poisson(mu=2).cdf(x).value == stats.poisson(2).pdf([1,20])).all(), 'poisson distribution cdf is not correct'
    # assert (abs(op.Poisson(mu=2).cdf(x).gradients - [derivative(stats.poisson(mu=2).cdf,1,dx=1e-6),derivative(stats.poisson(mu=2).cdf,20,dx=1e-6)])<1e-6).all(), 'poisson distribution cdf is not correct'

    # assert abs(op.Poisson(mu=2).pmf(4) - stats.poisson(2).pmf(4))<1e-6, 'poisson distribution pmf is not correct'
    # assert (op.Poisson(mu=2).pmf(x).value == stats.poisson(2).pdf([1,20])).all(), 'poisson distribution pmf is not correct'
    # assert (abs(op.Poisson(mu=2).pmf(x).gradients - [derivative(stats.poisson(mu=2).pmf,1,dx=1e-6),derivative(stats.poisson(mu=2).pmf,20,dx=1e-6)])<1e-6).all(), 'poisson distribution pmf is not correct'


    assert op.Poisson(mu=2).logpmf(4)
    assert op.Poisson(mu=2).logpmf(x)
    assert op.Poisson(mu=2).cdf(4)
    with warnings.catch_warnings(record=True) as w:
        op.Poisson(mu=2).cdf(x)
        assert len(w) > 0, "Poisson CDF does not display warning"
    assert op.Poisson(mu=2).logcdf(4)
    with warnings.catch_warnings(record=True) as w:
        op.Poisson(mu=2).logcdf(x)
        assert len(w) > 0, "Poisson CDF does not display warning"


def test_gammainc():
    assert op.gammainc(3, 2)
    with pytest.raises(TypeError):
        op.gammainc([3, 2], 3)


def test_Jacobian():
    inputs = [1, 2]
    seeds = [[1, 0], [0, 1]]
    def simple_func(x, y):
        return (x + y) ** 2
    assert (Jacobian(simple_func, inputs) == np.array([6.,6.])).all(), 'Jacobian is not correct'


def test_eq():
    x = FuncInput(np.array([0,3]),np.array([1]))
    y = FuncInput(np.array([0,3]),np.array([1]))
    assert x == y, 'equal function is not correct'

def test_neq():
    x = FuncInput(np.array([0,3]),np.array([1]))
    y = FuncInput(np.array([1,3]),np.array([1]))
    assert x != y, 'non-equal function is not correct'

def test_lt():
    x = FuncInput(np.array([0,3]),np.array([1]))
    y = FuncInput(np.array([1,4]),np.array([1]))
    assert x < y, 'less-than function is not correct'
    xreal = 3
    yreal = 4
    assert xreal < yreal, 'less-than function is not correct'

def test_gt():
    x = FuncInput(np.array([0,3]),np.array([1]))
    y = FuncInput(np.array([1,4]),np.array([1]))
    assert y > x, 'greater-than function is not correct'
    xreal = 3
    yreal = 4
    assert yreal > xreal, 'greater-than function is not correct'

def test_le():
    x = FuncInput(np.array([0,3]),np.array([1]))
    y = FuncInput(np.array([1,3]),np.array([1]))
    assert x <= y, 'less-or-equal-than function is not correct'
    xreal = 3
    yreal = 3
    assert xreal <= yreal, 'less-or-equal-tha function is not correct'

def test_ge():
    x = FuncInput(np.array([0,3]),np.array([1]))
    y = FuncInput(np.array([1,3]),np.array([1]))
    assert y >= x, 'greater-or-equal-than function is not correct'
    xreal = 3
    yreal = 3
    assert yreal >= xreal, 'greater-or-equal-than function is not correct'

def test_euclidean():
    x = FuncInput(np.array([1,2]), np.array([1,0]))
    y = FuncInput(np.array([4,5]), np.array([0,1]))
    result1 = op.euclidean(x, y)
    result2 = op.euclidean(1, y)
    result3 = op.euclidean(x, 2)
    expec_der1 = ((1-4) + (2-5))/distance.euclidean([1,2], [4,5])
    expec_der2 = ((1-4) + (0-5))/distance.euclidean([1,0], [4,5])
    expec_der3 = ((1-2) + (2-0))/distance.euclidean([1,2], [2,0])
    assert result1.value == distance.euclidean([1,2], [4,5]), 'euclidean function is wrong'
    assert (result1.gradients == np.array([expec_der1, -expec_der1])).all(), 'euclidean function is wrong'
    assert result2.value == distance.euclidean([1,0], [4,5]), 'euclidean function is wrong'
    assert (result2.gradients == np.array([0, -expec_der2])).all(), 'euclidean function is wrong'
    assert result3.value == distance.euclidean([1,2], [2,0]), 'euclidean function is wrong'
    assert (result3.gradients == np.array([expec_der3, 0])).all(), 'euclidean function is wrong'

# def test_validate_input():
#     x = FuncInput(np.array([1]),np.array([1,0]))
#     def func(x):
#         return x ** 2

#     assert func(x).value == op.validate_input(func)(x).value, 'validate input function is not correct'
#     assert (func(x).gradients == op.validate_input(func)(x).gradients).all(), 'validate input function is not correct'

# def test_validate_input():
#     x = FuncInput(np.array([1]),np.array([1,0]))
#     y = FuncInput(np.array([i]),np.array([0,1]))
#     def func(x,y):
#         return x+y
#     assert TypeError('Inputs must be type FuncInput or a real number')

# x = FuncInput(np.array([1]),np.array([1,0]))
# y = FuncInput(np.array([2]),np.array([0,1]))
# def func(x,y):
#     return x+y
# print(func(x,y).value == FuncInput.validate_input(func)(x,y).value)

# print(func == op.validate_input(func))
