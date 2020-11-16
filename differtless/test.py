import differtless.ad as ad
import numpy as np
import numbers
import pytest

def test_add():
    x1=ad.FuncInput(1)
    x2=ad.FuncInput(2)
    x3=ad.FuncInput(3)
    f = x1+x2+x3
    assert f.val == 6, "error with add"
    assert (f.ders == np.array([1,1,1])).all(), "error with add"