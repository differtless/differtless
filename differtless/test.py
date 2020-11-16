import differtless.ad as ad
import numpy as np
import pytest

def test_add():
    x1=ad.FuncInput(np.array([1]),np.array([1]))
    x1=ad.FuncInput(np.array([2]),np.array([1]))
    f = x1+x2
    assert f.val == 3, "error with add"
    assert (f.ders == np.array([1,1])).all(), "error with add"