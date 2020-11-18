import numpy as np
import pytest
import sys
sys.path.append('../')
from differtless.ad import FuncInput, preprocess
import differtless.operations as op

def test_add():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x + y
    assert f.val_ == 3, "add function is not correct"
    assert (f.ders_ == np.array([1,1])).all(), "add function is not correct"

def test_sub():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x - y
    assert f.val_ == -1, "sub function is not correct"
    assert (f.ders_ == np.array([1,-1])).all(), "sub function is not correct"

def test_mul():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x * y
    assert f.val_ == 2, "mul function is not correct"
    assert (f.ders_ == np.array([2,1])).all(), "mul function is not correct"

def test_truediv():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x / y
    assert f.val_ == 0.5, "truediv function is not correct"
    assert (f.ders_ == np.array([0.5,-0.25])).all(), "truediv function is not correct"

def test_floordiv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    y = FuncInput(np.array([-5]),np.array([0,1]))
    f = y // x
    assert f.val_ == -3, "floordiv function is not correct"
    assert (f.ders_ == np.array([1,0])).all(), "floordiv function is not correct"

def test_pow():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = x ** 3
    assert f.val_ == 8, "pow function is not correct"
    assert (f.ders_ == np.array([12,0])).all(), "pow function is not correct"

def test_neg():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = -x
    assert f.val_ == -2, "neg function is not correct"
    assert (f.ders_ == np.array([-1,0])).all(), "neg function is not correct"

def test_pos():
    x = FuncInput(np.array([-2]),np.array([1,0]))
    f = abs(x)
    assert f.val_ == 2, "pos function is not correct"
    assert (f.ders_ == np.array([-1,0])).all(), "pos function is not correct"

def test_abs():
    x = FuncInput(np.array([-2]),np.array([1,0]))
    f = abs(x)
    assert f.val_ == 2, "abs function is not correct"
    assert (f.ders_ == np.array([-1,0])).all(), "abs function is not correct"

def test_radd():
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = 1 + y
    assert f.val_ == 3, "radd function is not correct"
    assert (f.ders_ == np.array([0,1])).all(), "radd function is not correct"

def test_rsub():
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = 1 - y
    assert f.val_ == -1, "rsub function is not correct"
    assert (f.ders_ == np.array([0,-1])).all(), "rsub function is not correct"

def test_rmul():
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = 2 * y
    assert f.val_ == 4, "rmul function is not correct"
    assert (f.ders_ == np.array([0,2])).all(), "rmul function is not correct"

def test_rtruediv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = 1 / x
    assert f.val_ == 0.5, "rtruediv function is not correct"
    assert (f.ders_ == np.array([-0.25,0])).all(), "rtruediv function is not correct"

def test_rfloordiv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = -5 // x
    assert f.val_ == -3, "rfloordiv function is not correct"
    assert (f.ders_ == np.array([1,0])).all(), "rfloordiv function is not correct"

def rpow():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = 2**x
    assert f.val_ == 4, "rpow function is not correct"
    assert (abs(f.ders_ - np.array([2.77258872,0]))<1e-6).all(), "rpow function is not correct"

def test_exp():
    x = FuncInput(np.array([0]),np.array([1,0]))
    f = op.exp(x)
    assert f.val_ == 1, "exp function is not correct"
    assert (f.ders_ == np.array([1,0])).all(), "exp function is not correct"

def test_expm1():
    x = FuncInput(np.array([0]),np.array([1,0]))
    f = op.expm1(x)
    assert f.val_ == 0, "expm1 function is not correct"
    assert (f.ders_ == np.array([1,0])).all(), "expm1 function is not correct"

def test_exp2():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = op.exp2(x)
    assert f.val_ == 4, "exp2 function is not correct"
    assert (abs(f.ders_ - np.array([2.77258872,0]))<1e-6).all(), "exp2 function is not correct"

def test_log():
    x = FuncInput(np.array([3]),np.array([1,0]))
    f = op.log(x)
    assert (abs(f.val_ - 1.09861229) < 1e-6).all(), "log function is not correct"
    assert (abs(f.ders_ - np.array([0.33333333,0]))<1e-6).all(), "log function is not correct"

def test_log10():
    x = FuncInput(np.array([3]),np.array([1,0]))
    f = op.log10(x)
    assert (abs(f.val_ - 0.47712125) < 1e-6).all(), "log10 function is not correct"
    assert (abs(f.ders_ - np.array([0.14476483,0]))<1e-6).all(), "log10 function is not correct"

def test_log2():
    x = FuncInput(np.array([3]),np.array([1,0]))
    f = op.log2(x)
    assert (abs(f.val_ - 1.5849625) < 1e-6).all(), "log2 function is not correct"
    assert (abs(f.ders_ - np.array([0.48089835,0]))<1e-6).all(), "log2 function is not correct"

def test_log1p():
    x = FuncInput(np.array([0]),np.array([1,0]))
    f = op.log1p(x)
    assert f.val_ == 0, "log1p function is not correct"
    assert (f.ders_ == np.array([1,0])).all(), "log1p function is not correct"

def test_logaddexp():
    x = FuncInput(np.array([0]),np.array([1,0]))
    y = FuncInput(np.array([1]),np.array([0,1]))
    f = op.logaddexp(x,y)
    assert (abs(f.val_ - 1.31326169) < 1e-6).all(), "logaddexp function is not correct"
    assert (abs(f.ders_ - np.array([0.26894142,0.73105858]))<1e-6).all(), "logaddexp function is not correct"

def test_logaddexp2():
    x = FuncInput(np.array([2]),np.array([1,0]))
    y = FuncInput(np.array([1]),np.array([0,1]))
    f = op.logaddexp2(x,y)
    assert (abs(f.val_ - 2.32192809) < 1e-6).all(), "logaddexp2 function is not correct"
    assert (abs(f.ders_ - np.array([1.15415603,0.57707802]))<1e-6).all(), "logaddexp2 function is not correct"

def test_sin():
    x = FuncInput(np.array([np.pi/6]),np.array([1,0]))
    f = op.sin(x)
    assert (abs(f.val_ - 0.5)<1e-6).all(), "sin function is not correct"
    assert (abs(f.ders_ - np.array([0.8660254,0.]))<1e-6).all(), "sin function is not correct"

def test_cos():
    x = FuncInput(np.array([np.pi/3]),np.array([1,0]))
    f = op.cos(x)
    assert (abs(f.val_ - 0.5)<1e-6).all(), "cos function is not correct"
    assert (abs(f.ders_ - np.array([-0.8660254,0.]))<1e-6).all(), "cos function is not correct"

def test_tan():
    x = FuncInput(np.array([np.pi/3]),np.array([1,0]))
    f = op.tan(x)
    assert (abs(f.val_ - 1.73205081) < 1e-6).all(), "tan function is not correct"
    assert (abs(f.ders_ - np.array([4.,0.]))<1e-6).all(), "tan function is not correct"

    
def test_arcsin():
    assert NotImplementedError('Function not yet implemented in differtless')


def test_preprocess():
    inputs_1 = [1, 2]
    seed_1 = [[1,1],[2,2]]
    assert preprocess(inputs_1) == [FuncInputs(np.array([1]), np.array([1,0])), FuncInputs(np.array([2]), np.array([0,1]))], 'preprocess is mishandling seed = []'
    assert preprocess(inputs_1, seed_1) == [FuncInputs(np.array([1]), np.array([1,1])), FuncInputs(np.array([2]), np.array([2,2]))], 'preprocess is not creating correct gradients'




