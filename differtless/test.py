# testing

# from differtless import ad
# from differtless import operations as op
import numpy as np

# # inputs = [[5,2,3], [2,3,4]]
# inputs = [[2,3,4],[3,2,1]]
# # inputs = [3,2]
# def f(x):
#     return 7 * op.sin(x) + 6
#
# def f1(x1, x2):
#     return (x1 * x2) + x1
# def f2(x1, x2):
#     return x1/x2
#
# result = ad.Jacobian([f1, f2], inputs)
# print(result)
# x = ad.preprocess(inputs)[0]
# print(x)
# abs(x)
# print(x)


from differtless.ad import FuncInput
import numpy as np

x = FuncInput(np.array([1]),np.array([1,0]))
y = FuncInput(np.array([2]),np.array([0,1]))
print(x * y)