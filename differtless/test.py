# testing

# from differtless import ad
# from differtless import operations as op
# from ad import *
# import numpy as np
#
# # # inputs = [[5,2,3], [2,3,4]]
# inputs = [[2,3,4],[3,2,1]]
# # inputs = [3,2]
# def f(x):
#     return 7 * op.sin(x) + 6
#
# def f1(x1, x2):
#     return x1 / x2
# def f2(x1, x2):
#     return x1/x2
#
# result = forward(f1, inputs)
# print(result)
# x = ad.preprocess(inputs)[0]
# print(x)
# abs(x)
# print(x)


from ad import *
import numpy as np

inputs = [1, 2]
seeds = [[1, 0], [0, 1]]
def simple_func(x, y):
    return (x + y) ** 2
print(forward(simple_func, inputs, seeds).value)
