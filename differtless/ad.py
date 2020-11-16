"""

Auto-Differentiation Module:

Contents:
    - Preprocessing function
    - FuncInput class
    - forward AD function
    - Minimize function
"""

"""
Preprocessing Function
Allows us to deal with scalars and vectors
Takes in a list or numpy.array of inputs (1 x N) and an optional matrix seeds (N x N)
If the user inputs a scalar it will be converted to a 1 x 1 vector
seeds defaults to None, in which case we will use an N x N identity matrix where N = len(inputs)
For each value in inputs (and if the inputs are vectors, each component in each vector) and row in seeds we will instantiate the FuncInput object described below


check input for - real number, check that derivatives seeds are the same length (square matrix)
raise TypeErrors/ ArgumentErrors
"""
import numbers 
import numpy as np

def preprocess(inputs, seeds = []): 
  """
  Takes in: 
  - inputs:  list of N inputs- either scalar or vector
  - seeds (optional): NxN matrix seeds. 
      - default is N-sized ID matrix

  Checks: 
  - all elements of inputs and seeds must be real numbers
  - seeds must be an NxN matrix (if inputted)

  Returns: list of N FuncInput's, where the i'th FuncInput corresponds to the 
  i'th input and the i'th seed derivatives
  """

  N = len(inputs)
  for element in inputs: 
    if not isinstance(element, numbers.Real): 
      for e in element: 
        if not isinstance(e, numbers.Real):
          raise TypeError("Please make sure all inputs are Real Numbers") 


  if (seeds == []): 
    # if seeds = [], make ID matrix
    for i in range(N): 
      new_row = []
      for j in range(N):
        if (i==j):
          new_row.append(1)
        else: 
          new_row.append(0)
      seeds.append(new_row) 

  else:
    # check if NXN matrix 
    len_seeds = len(seeds)
    if (len_seeds != N): 
      raise ValueError("Make sure your seeds matrix is the right size")
    else:
      for row in seeds:
        if (len(row) !=N):
          raise ValueError("Make sure your seeds matrix is the right size")
        for element in row: 
          if not isinstance(element, numbers.Real): 
            raise TypeError("Please make sure all inputs are Real Numbers") 
  
  new_inputs = []
  # make scalar values and tuples into lists for inputs
  for val in inputs:
    if (isinstance(val, numbers.Real)):
      new_inputs.append([val])
    elif (isinstance(val, list)):
      new_inputs.append(val)
    elif (isinstance(val, tuple)):
      holder = []
      for i in val: 
        holder.append(i)
      new_inputs.append(holder)

  r = []
  for i in range(N): 
    r.append(FuncInput(new_inputs[i], seeds[i]))
    # r.append((new_inputs[i], seeds[i]))

  return r


#Testing for Preprocessing
# inputs = [1,2,3]
# seeds = [[42, 1, 1], [2, 42, 2], [3, 3, 42]]

# print(preprocess(inputs, seeds))
# # #  = [FuncInput([1], [42, 1, 1]), FuncInput([2], [2, 42, 2]), FuncInput([3], [3, 3, 42])] 

# inputs = [(1,2),2,3]
# seeds = [[42, 1, 1], [2, 42, 2], [3, 3, 42]]

# print(preprocess(inputs, seeds))

# # inputs = [(1,2),2,3]
# # seeds = [[42, 1, 1], [2, 42], [3, 3, 42]]


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
        Overwritten unary dunder methods: __neg__, __abs__

    EXAMPLE
    ========
    >>> x = FuncInput(np.array([1]), np.array([1, 0]))
    FuncInput([2], [1 0])
    >>> print(x)
    FuncInput object with value [2] and gradients [1 0] with respect to each input
    >>> y = FuncInput(np.array([3]), np.array([0, 1]))
    FuncInput([3], [0 1])
    >>> x + y
    FuncInput([5], [1, 1])
    >>> x * y
    FuncInput([6], [3 2])
    >>> 2 * x + y
    FuncInput([7], [2 1])

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
            if not isinstance(other, FuncInput) and not isinstance(other, numbers.Real):
                raise TypeError('Inputs must be type FuncInput or a real number')
            return func(self, other)
        return wrapper



    ## Overwritten basic functions ##

    # Addition
    @validate_input
    def __add__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ + other.val_
            new_ders = self.ders_ + other.ders_
        else:
            new_val = self.val_ + other
            new_ders = self.ders_

        return FuncInput(new_val, new_ders)

    # Subtraction
    @validate_input
    def __sub__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ - other.val_
            new_ders = self.ders_ - other.ders_
        else:
            new_val = self.val_ - other
            new_ders = self.ders_

        return FuncInput(new_val, new_ders)

    # Multiplication
    @validate_input
    def __mul__(self, other):
        if isinstance(other, FuncInput):
            new_val = self.val_ * other.val_
            new_ders = self.val_ * other.ders_ + self.ders_ * other.val_
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
            new_ders = quot_rule(self.val_, other.val_, self.ders_, other.ders_)
        else:
            new_val = self.val_ / other
            new_ders = quot_rule(self.val_, other, self.ders_, 0)

        return FuncInput(new_val, new_ders)

    # floor Division
    @validate_input
    def __floordiv__(self, other):
        def floor_quot_rule(high,low,dhigh,dlow): return ((low * dhigh) - (high * dlow))//(low ** 2)

        if isinstance(other, FuncInput):
            new_val = self.val_ // other.val_
            new_ders = floor_quot_rule(self.val_, other.val_, self.ders_, other.ders_)
        else:
            new_val = self.val_ // other
            new_ders = floot_quot_rule(self.val_, other, self.ders_, other)


        return FuncInput(new_val, new_ders)

    # Exponentiation
    @validate_input
    def __pow__(self, other):
        def pow_rule(x, exp, dx): return (exp * (x ** (exp - 1))) * dx

        if isinstance(other, FuncInput):
            new_val = self.val_ ** other.val_
            new_ders = pow_rule(self.val_, other.val_, self.ders_)
        else:
            new_val = self.val_ ** other
            new_ders = pow_rule(self.val_, other, self.ders_)

        return FuncInput(new_val, new_ders)

    ## Unary operations ##

    # Negate
    def __neg__(self):
        new_vals = -self.val_
        new_ders = -self.ders_
        return FuncInput(new_vals, new_ders)

    # Positive
    def __pos__(self):
        new_vals = abs(self.val_)
        new_ders = abs(self.ders_)
        return FuncInput(new_vals, new_ders)

    # Absolute value
    def __abs__(self):
        new_vals = abs(self.val_)
        new_ders = abs(self.ders_)
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
            new_ders = -(other * self.ders_)
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    # Reverse floor division
    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other // self.val_
            new_ders = -(other * self.ders_)
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    # Reverse power
    def __rpow__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other ** self.val_
            new_ders = np.log(other) * new_val * self.ders_
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')
