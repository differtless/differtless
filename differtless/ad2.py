"""

Auto-Differentiation Module:

Contents:
    - Preprocessing function
    - FuncInput class
    - forward AD function
    - Minimize function

"""
import numbers
import numpy as np


class FuncInput():
    """

    Class to represent the inputs to forward mode of automatic differentiation.

    ATTRIBUTES
    ==========
        val_ : np.array()
            NumPy array containing the value(s) of the input
        ders_ : np.array()
            NumPy array containing the value(s) of the gradient of the input with respect to all inputs

    METHODS
    ========
        Overwritten basic operation dunder methods: __add__, __sub__, __mul__, __truediv__, __floordiv__, and __pow__ as well as the their reverse counter-parts.
        All operations are pairwise by component.
        Overwritten unary dunder methods: __neg__

    EXAMPLE
    ========
    >>> ex = FuncInput(np.array([1]), np.array([1, 0, 0]))
    FuncInput([1], [1 0 0])
    >>> print(ex)
    FuncInput object with value [1] and gradients [1 0 0] with respect to each input

    """

    def __init__(self, value, seed):
        self.val_ = value
        self.ders_ = seed

    def __str__(self):
        return f'FuncInput object with value {self.val_} and gradients {self.ders_} with respect to each input'

    def __repr__(self):
        return f'FuncInput({self.val_}, {self.ders_})'


    # Wrapper that will make sure all inputs are type FuncInput or a real number
    # def validate_input(func):
    #     def wrapper(self, other):
    #         if not isinstance(other, FuncInput) or not isinstance(other, numbers.Real):
    #             raise TypeError('Inputs must be type FuncInput or a real number')
    #         return func(self, other)
    #     return wrapper



    ## Overwritten basic functions ##

    # Addition
    # @validate_input
    def __add__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ + other.val_
            new_ders = [self.ders_[i] + other.ders_[i] for i in range(len(self.ders_))]
        else:
            new_val = self.val_ + other
            new_ders = self.ders_

        return FuncInput(new_val, new_ders)
