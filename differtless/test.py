from ad import FuncInput
import operations as op
import numpy as np
import pytest

def test_add():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x + y
    assert f.val_ == 3, "Add function is not correct"
    assert (f.ders_ == np.array([1,1])).all(), "Add function is not correct"

def test_sub():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x - y
    assert f.val_ == -1, "Sub function is not correct"
    assert (f.ders_ == np.array([1,-1])).all(), "Sub function is not correct"

def test_mul():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x * y
    assert f.val_ == 2, "Mul function is not correct"
    assert (f.ders_ == np.array([2,1])).all(), "Mul function is not correct"

def test_truediv():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x / y
    assert f.val_ == 0.5, "Truediv function is not correct"
    assert (f.ders_ == np.array([0.5,-0.25])).all(), "Truediv function is not correct"

def test_floordiv():
    x = FuncInput(np.array([2]),np.array([1,0]))
    y = FuncInput(np.array([-5]),np.array([0,1]))
    f = y // x
    assert f.val_ == -3, "Floordiv function is not correct"
    assert (f.ders_ == np.array([1,0])).all(), "Floordiv function is not correct"

def test_pow():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = x ** 3
    assert f.val_ == 8, "Pow function is not correct"
    assert (f.ders_ == np.array([12,0])).all(), "Pow function is not correct"

def test_neg():
    x = FuncInput(np.array([2]),np.array([1,0]))
    f = -x
    assert f.val_ == -2, "Neg function is not correct"
    assert (f.ders_ == np.array([-1,0])).all(), "Neg function is not correct"

def test_pos():
    x = FuncInput(np.array([-2]),np.array([1,0]))
    f = abs(x)
    assert f.val_ == 2, "Pos function is not correct"
    assert (f.ders_ == np.array([-1,0])).all(), "Pos function is not correct"

def test_abs():
    x = FuncInput(np.array([-2]),np.array([1,0]))
    f = abs(x)
    assert f.val_ == 2, "Abs function is not correct"
    assert (f.ders_ == np.array([-1,0])).all(), "Abs function is not correct"

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
    assert f.val_ == 4, "rfloordiv function is not correct"
    assert np.linalg.norm(f.ders_ - np.array([2.77258872,0]))<1e-6.all(), "rfloordiv function is not correct"

# y = FuncInput(np.array([2]),np.array([0,2]))
# f = 3 - y
# print(f)
# x = FuncInput(np.array([2]),np.array([1,0]))
# print(2**x)
# print(np.linalg.norm(np.array([2.7725887,0]) - np.array([2.77258872,0]))<1e-6)
# print(1/(2*x))
# print(1/(x*x))