from ad import FuncInput
import operations as op
import numpy as np
import pytest

def test_add():
    x1 = FuncInput(np.array([1]),np.array([1,0,0]))
    x2 = FuncInput(np.array([2]),np.array([0,1,0]))
    x3 = FuncInput(np.array([3]),np.array([0,0,1]))
    f = x1 + x2 + x3
    assert f.val_ == 6, "Add function is not correct"
    assert (f.ders_ == np.array([1,1,1])).all(), "Add function is not correct"


def test_sub():
    x1 = FuncInput(np.array([1]),np.array([1,0,0]))
    x2 = FuncInput(np.array([2]),np.array([0,1,0]))
    x3 = FuncInput(np.array([3]),np.array([0,0,1]))
    f = x1 - x2 - x3
    assert f.val_ == -4, "Sub function is not correct"
    assert (f.ders_ == np.array([1,-1,-1])).all(), "Sub function is not correct"
