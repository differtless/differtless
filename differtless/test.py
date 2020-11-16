import differtless.ad as ad
import numpy as np
import numbers
import pytest

def test_add():
    x1=1
    x2=2
    x3=3
    f = x1+x2+x3
    assert f.x == 6, "error with add"
    assert (f.dx == np.array([1,1,1])).all(), "error with add"