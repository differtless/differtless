import differtless.ad as ad

def test_add():
    fn=ad.FuncInput(3)
    x1=fn.create_variable(4)
    x2=fn.create_variable(5)
    x3=fn.create_variable(6)
    f = x1+x2+x3
    assert f.x == 15, "error with add"
    assert (f.dx == np.array([1,1,1])).all(), "error with add"