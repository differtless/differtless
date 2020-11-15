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
    def validate_input(func):
        def wrapper(self, other):
            if not isinstance(other, FuncInput) or not isinstance(other, numbers.Real):
                raise TypeError('Inputs must be type FuncInput or a real number')
            return func(self, other)
        return wrapper



    ## Overwritten basic functions ##

    # Addition
    @validate_input
    def __add__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ + other.val_
            new_ders = [self.ders_[i] + other.ders_[i] for i in range(len(self.ders_))]
        else:
            new_val = self.val_ + other
            new_ders = self.ders_

        return FuncInput(new_val, new_ders)

    # Subtraction
    @validate_input
    def __sub__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ - other.val_
            new_ders = [self.ders_[i] - other.ders_[i] for i in range(len(self.ders_))]
        else:
            new_val = self.val_ - other
            new_ders = self.ders_

        return FuncInput(new_val, new_ders)

    # Multiplication
    @validate_input
    def __mul__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ * other.val_
            new_ders = [(self.val_ * other.ders_[i]) + (self.ders_[i] * other.val_)for i in range(len(self.ders_))]
        else:
            new_val = self.val_ * other
            new_ders = [self_der * other for self_der in self.ders_]

        return FuncInput(new_val, new_ders)

    # True Division
    @validate_input
    def __truediv__(self, other):
        def quot_rule(high,low,dhigh,dlow): return ((low * dhigh) - (high * dlow))/(low ** 2)

        if isinstance(other, FuncInput):
            new_val = self.val_ / other.val_
            new_ders = [quot_rule(self.val_[0], other.val_[0], self.ders_[i], other.ders_[i]) for i in range(len(self.ders_))]
        else:
            new_val = self.val_ / other
            new_ders = [quot_rule(self.val_[0], other, self_der, 0) for self_der in self.ders_]

        return FuncInput(new_val, new_ders)

    # floor Division
    @validate_input
    def __floordiv__(self, other):
        def floor_quot_rule(high,low,dhigh,dlow): return ((low * dhigh) - (high * dlow))//(low ** 2)

        if isinstance(other, FuncInput):
            new_val = self.val_ // other.val_
            new_ders = [floor_quot_rule(self.val_[0], other.val_[0], self.ders_[i], other.ders_[i]) for i in range(len(self.ders_))]
        else:
            new_val = self.val_ // other
            new_ders = [floor_quot_rule(self.val_[0], other, self_der, 0) for self_der in self.ders_]

        return FuncInput(new_val, new_ders)

    # Exponentiation
    @validate_input
    def __pow__(self, other):
        def pow_rule(x, exp, dx): return (exp * (x ** (exp - 1))) * dx

        if isinstance(other, FuncInput):
            new_val = self.val_ ** other.val_
            new_ders = [pow_rule(self.val_[0], other.val_[0], self_der) for self_der in self.ders_]
        else:
            new_val = self.val_ ** other
            new_ders = [pow_rule(self.val_[0], other, self_der) for self_der in self.ders_]

        return FuncInput(new_val, new_ders)

    ## Unary operations ##
    def __neg__(self):
        new_vals = -self.val_
        new_ders = -self.ders_
        return FuncInput(new_vals, new_ders)


    ## Reverse commutative operations ##

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__


    ## Non-commutative reverse operations ##

    # Reverse true division
    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other / self.val_
            new_ders = [-(other * self_der) for self_der in self.ders_]
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    # Reverse floor division
    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other // self.val_
            new_ders = [-(other * self_der) for self_der in self.ders_]
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    # Reverse power
    def __rpow__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other ** self.val_
            new_ders = np.zeros(len(self.ders_[0]))
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')
