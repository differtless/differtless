from ad import FuncInput
import operations as op
import numpy as np
import pytest

def test_add():
    x1 = FuncInput(np.array([1]),np.array([1,0]))
    x2 = FuncInput(np.array([2]),np.array([0,1]))
    f = x1 + x2
    assert f.val_ == 3, "Add function is not correct"
    assert (f.ders_ == np.array([1,1])).all(), "Add function is not correct"

# def test_exp():
#     x=FuncInput(np.array([1]),np.array([1]))
#     f = op.exp(x)
#     assert f.val_ == 1, "error with add"
#     assert (f.ders_ == 1).all(), "error with add"

# x1 = FuncInput(np.array([1]),np.array([1,0]))
# x2 = FuncInput(np.array([2]),np.array([0,1]))
# f = x1 + x2

# x = FuncInput(np.array([1]),np.array([1]))
# print(isinstance(x, FuncInput))
# f = op.exp(x)
# print(f)

