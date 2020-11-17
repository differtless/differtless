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
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x + y
    assert f.val_ == 3, "rAdd function is not correct"
    assert (f.ders_ == np.array([1,1])).all(), "rAdd function is not correct"

def test_rsub():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x - y
    assert f.val_ == -1, "rSub function is not correct"
    assert (f.ders_ == np.array([1,-1])).all(), "rSub function is not correct"

def test_rmul():
    x = FuncInput(np.array([1]),np.array([1,0]))
    y = FuncInput(np.array([2]),np.array([0,1]))
    f = x * y
    assert f.val_ == 2, "rMul function is not correct"
    assert (f.ders_ == np.array([2,1])).all(), "rMul function is not correct"

x = FuncInput(np.array([-2]),np.array([1,0]))
f = abs(x)


print(f)