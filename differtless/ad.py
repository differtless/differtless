"""
Auto-Differentiation Module:
Contents:
    - Preprocessing function
    - FuncInput class
    - forward AD function
    - minimize function
"""

import numbers
import numpy as np
from scipy.optimize import minimize as spmin # needed for Minimize function
# prettify prints (no scientific notation)
np.set_printoptions(suppress=True)

def preprocess(inputs, seeds = []):
    """
    Function that produces a list of FuncInput objects with respect to each input.
    To be used within forward() to process inputs.

    PARAMETERS
    ==========
      inputs : iterable type (list, np.array(), etc.)
          Iterable containing the input values to the functions
      seeds : iterable type (list, np,array(), etc.)
          Iterable containing the gradients of each input with respect to all other inputs (default is [])
    RETURNS
    =======
      A list of FuncInput objects with the appropriate gradients (if no seed is
      specified the gradients are assigned to be unit vectors)
    EXAMPLE
    ========
    >>> inputs = [1, 2]
    >>> seeds = [[1, 0], [0, 1]]
    >>> preprocess(inputs, seeds)
    [FuncInput([1], [1 0]), FuncInput([2], [0 1])]
    """
    check_num = isinstance(inputs, numbers.Real)
    if check_num: 
        hold = [].append(inputs)
        inputs = hold
    
    N = len(inputs)
    for element in inputs:
        if not isinstance(element, numbers.Real):
            for e in element:
                if not isinstance(e, numbers.Real):
                    raise TypeError("Please make sure all inputs are Real Numbers")                    


    if seeds == []:
        # if seeds = [], make ID matrix
        new_seeds = []
        for i in range(N):
            new_row = []
            for j in range(N):
                if (i==j):
                    new_row.append(1)
                else:
                    new_row.append(0)
            new_seeds.append(new_row)
    else:
        
        if (isinstance(seeds, numbers.Real)): 
            hold = [[]][0].append(seeds)
            seeds = hold
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

        # make seed rows into np.arrays
        new_seeds = []
        for row in seeds:
            new_seeds.append(np.array(row))

    new_inputs = []
    # make scalar values and tuples into np.arrays for inputs
    for val in inputs:
        if (isinstance(val, numbers.Real)):
            new_inputs.append(np.array([val]))
        elif (isinstance(val, list)):
            new_inputs.append(np.array(val))
        elif (isinstance(val, tuple)):
            holder = []
            for i in val:
                holder.append(i)
            new_inputs.append(np.array(holder))

    r = []
    for i in range(N):
        r.append(FuncInput(new_inputs[i], new_seeds[i]))

    return r



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
        Overwritten basic operation dunder methods: __add__, __sub__, __mul__,
        __truediv__, __floordiv__, and __pow__ as well as the their reverse counter-parts.
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
        value = self.value
        gradient = self.gradients

        return f'Value:\n {value}\nGradient(s):\n {gradient}'

    def __repr__(self):
        return f'FuncInput({self.value}, {self.gradients})'

    @property
    def value(self):
        return np.squeeze(self.val_)

    @property
    def gradients(self):
        return np.squeeze(self.ders_)


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
            new_ders = [self.val_ * other.ders_[i] + self.ders_[i] * other.val_ for i in range(len(self.ders_))]
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
            new_ders = [quot_rule(self.val_, other.val_, self.ders_[i], other.ders_[i]) for i in range(len(self.ders_))]
        else:
            new_val = self.val_ / other
            new_ders = [quot_rule(self.val_, other, self_der, 0) for self_der in self.ders_]

        return FuncInput(new_val, new_ders)

    # floor Division
    @validate_input
    def __floordiv__(self, other):
        def floor_quot_rule(high,low,dhigh,dlow): return ((low * dhigh) - (high * dlow))//(low ** 2)

        if isinstance(other, FuncInput):
            new_val = self.val_ // other.val_
            new_ders = [floor_quot_rule(self.val_, other.val_, self.ders_[i], other.ders_[i]) for i in range(len(self.ders_))]
        else:
            new_val = self.val_ // other
            new_ders = [floot_quot_rule(self.val_, other, self_der, 0) for self_der in self.ders_]


        return FuncInput(new_val, new_ders)

    # Exponentiation
    def __pow__(self, other):
        def pow_rule(x, exp, dx, dexp):
            if (x == 0).any():
                return 0

            return (x ** exp) * (((exp * dx)/x) + dexp*np.log(x))

        if isinstance(other, FuncInput):
            # check for negative bases in the case of even powers, do this iteratively for VVFs
            self.val_ = np.array([abs(self_val) if other.val_[i]%2 == 0 else self_val for i, self_val in enumerate(self.val_)])

            new_val = self.val_ ** other.val_

            new_ders = [pow_rule(self.val_, other.val_, self.ders_[i], other.ders_[i]) for i in range(len(self.ders_))]
        else:
            # check for negative bases in the case of even powers
            self = abs(self) if other%2 == 0 else self
            new_val = self.val_ ** other
            new_ders = [pow_rule(self.val_, other, self.ders_[i], 0) for i in range(len(self.ders_))]

        return FuncInput(new_val, new_ders)

    ## Comparison Operations ##
    def __eq__(self, other):
        if isinstance(other, FuncInput):
            return self.val_ == other.val_ and self.ders_ == other.ders_
        else:
            raise ValueError('Cannot compare FuncInput to non-FuncInput')

    def __neq__(self, other):
        if isinstance(other, FuncInput):
            return self.val_ != other.val_ or self.ders_ != other.ders_
        else:
            raise ValueError('Cannot compare FuncInput to non-FuncInput')

    @validate_input
    def __lt__(self, other):
        if isinstance(other, FuncInput):
            return (self.val_ < other.val_).all()
        elif isinstance(other, numbers.Real):
            return (self.val_ < other).all()

    @validate_input
    def __gt__(self, other):
        if isinstance(other, FuncInput):
            return (self.val_ > other.val_).all()
        elif isinstance(other, numbers.Real):
            return (self.val_ > other).all()

    @validate_input
    def __le__(self, other):
        if isinstance(other, FuncInput):
            return (self.val_ <= other.val_).all()
        elif isinstance(other, numbers.Real):
            return (self.val_ <= other).all()

    @validate_input
    def __ge__(self, other):
        if isinstance(other, FuncInput):
            return (self.val_ >= other.val_).all()
        elif isinstance(other, numbers.Real):
            return (self.val_ >= other).all()

    ## Unary operations ##

    # Negate
    def __neg__(self):
        new_val = -self.val_
        new_ders = [-self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)

    # Positive
    def __pos__(self):
        return self

    # Absolute value
    def __abs__(self):
        new_val = np.abs(self.val_)
        new_ders = np.abs(self.ders_) if (self.val_ > 0).any() else [-self_der for self_der in self.ders_]
        return FuncInput(new_val, new_ders)

    ## Reverse commutative operations ##
    __radd__ = __add__
    __rmul__ = __mul__

    ## Non-commutative reverse operations ##

    @validate_input
    def __rsub__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other -self.val_
            new_ders = [-self_der for self_der in self.ders_]
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    # Reverse true division
    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other / self.val_
            new_ders = [-other * self.ders_[i] / self.val_ ** 2 for i in range(len(self.ders_))]
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    # Reverse floor division
    def __rfloordiv__(self, other):
        if isinstance(other, numbers.Real):
            new_val = other // self.val_
            new_ders = [-other * self.ders_[i] // self.val_ ** 2 for i in range(len(self.ders_))]
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')

    def __rpow__(self, other):
        def pow_rule(x, exp, dx, dexp):
            if x == 0:
                return 0
            else:
                return (x ** exp) * (((exp * dx)/x) + dexp*np.log(x))

        if isinstance(other, numbers.Real):
            new_val = other ** self.val_
            new_ders = [pow_rule(other, self.val_, 0, self.ders_[i]) for i in range(len(self.ders_))]
            return FuncInput(new_val, new_ders)
        else:
            raise TypeError('Inputs must be FuncInput or real numbers')


def forward(funs, inputs, seeds = []):
    """
    Function that executes forward mode of automatic differentiation. Executes a
    pre-defined function while keeping track of the gradients of the inputs with
    respect to all other inputs

    PARAMETERS
    ==========
        fun:
            Pre-defined function, or list of functions, to be executed
        inputs : iterable type (list, np.array(), etc.)
            Iterable containing the input values to the functions
        seeds : iterable type (list, np,array(), etc.)
            Iterable containing the gradients of each input with respect to all other inputs (default is [])
    ACTIONS
    =======
        Preprocesses inputs to FuncInput type
    RETURNS
    =======
        Results of pre-defined function: updated values and gradients
    EXAMPLE
    ========
    >>> inputs = [1, 2]
    >>> seeds = [[1, 0], [0, 1]]
    >>> def simple_func(x, y):
    ...     return (x + y) ** 2
    >>> forward(simple_func, inputs, seeds)
    FuncInput([9], [6. 6.])
    """

    func_inputs = preprocess(inputs, seeds)

    # if multiple functions, run them all and stack the results
    try:
        result_val = []
        result_grad = []

        for fun in funs:

            output = fun(*func_inputs)
            out_val = output.val_
            out_grad = output.ders_

            result_val.append(out_val)


            for i, val in enumerate(out_grad):
                if not isinstance(val, numbers.Real):
                    # if function is single value or all values are the same
                    if len(val) == 1 or (val == np.min(val)).all():
                        out_grad[i] = val[0]
            out_grad = np.array(out_grad)
            result_grad.append(out_grad)

        result_grad = np.squeeze(np.array(result_grad))
        result_val = np.squeeze(np.array(result_val))
        return FuncInput(result_val, result_grad)

    except TypeError:

        output = funs(*func_inputs)
        out_val = output.val_
        out_grad = output.ders_

        for i, val in enumerate(out_grad):
            # if function is single value or all values are the same
            if not isinstance(val, numbers.Real):
                if len(val) == 1 or (val == np.min(val)).all():
                    out_grad[i] = val[0]

        out_grad = np.array(out_grad)

        out_val = np.squeeze(np.array(out_val))
        out_grad = np.squeeze(np.array(out_grad))
        return FuncInput(out_val, out_grad)


def Jacobian(funs, inputs):
    """
    Function that executes forward mode of automatic differentiation. Executes a
    pre-defined function while keeping track of the gradients of the inputs with
    respect to all other inputs

    PARAMETERS
    ==========
        fun:
            Pre-defined function, or list of functions, to be executed
        inputs : iterable type (list, np.array(), etc.)
            Iterable containing the input values to the functions
    ACTIONS
    =======
        - Preprocesses inputs to FuncInput type
        - Execute forward mode with inputs and default seed (identity)
    RETURNS
    =======
        Return the resulting gradients from forward mode which will be the Jacobian
    EXAMPLE
    ========
    >>> inputs = [1, 2]
    >>> seeds = [[1, 0], [0, 1]]
    >>> def simple_func(x, y):
    ...     return (x + y) ** 2
    >>> forward(simple_func, inputs, seeds)
    FuncInput([9], [6. 6.])
    """

    result = forward(funs, inputs, [])
    return result.gradients


def minimize(fun, x0, descriptive=False, args=(), method=None, hess=None, hessp=None, bounds=None, 
             constraints=(), tol=None, callback=None, options=None):
    """
    Wrapper for scipy.optimize.minimize that automatically uses differtless to feed in the Jacobian.

    PARAMETERS
    ==========
        fun : callable
            Pre-defined function, or list of functions, to be executed
        x0 : ndarray, shape (n,)
            Initial guess. Array of real elements of size (n,),
            where 'n' is the number of independent variables.
        descriptive : Bool
            If "True", returns full scipy `OptimizeResult`.
            If "False", returns only the solution array.
        args : tuple, optional
            Same as for scipy.optimize.minimize
            Extra arguments passed to the objective function and its derivatives.
        method : str or callable, optional
            Same as for scipy.optimize.minimize
            Type of solver. If not given, chosen to be one of
            BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds.
        hess: {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
            Same as for scipy.optimize.minimize
            Method for computing the Hessian matrix.
        hessp: callable, optional
            Same as for scipy.optimize.minimize
            Hessian of objective function times an arbitrary vector p.
        bounds: sequence or `Bounds`, optional
            Same as for scipy.optimize.minimize
            Bounds on variables.
        constraints: {Constraint, dict} or List of {Constraint, dict}, optional
            Same as for scipy.optimize.minimize
            Constraints definition.
        tol: float, optional
            Same as for scipy.optimize.minimize
            Tolerance for termination.
        options: dict, optional
            Same as for scipy.optimize.minimize
            A dictionary of solver options.
    ACTIONS
    =======
        - Makes function definition compatible with scipy and uses differtless to calculate Jacobian.
        - Performs optimization using scipy.optimize.minimize
    RETURNS
    =======
        If descriptive == False, returns the solution array.
        If descriptive == True, returns scipy `OptimizeResult` object.    
    EXAMPLE
    ========
    >>> guess = [1, 2]
    >>> def simple_func(x, y):
    ...     return 1/(x*y)
    >>> minimize(simple_func, guess)
    array([59.08683257, 44.60727855])
    """

    def fun_flat(args):
        '''Makes `fun` compatible with scipy, allowing for input of arguments as a single parameter.'''
        return fun(*args)

    def jac(x):
        '''Uses differtless to return Jacobian.'''
        return Jacobian(fun, x)

    # Call scipy.optimize.minimize to perform optimization
    optim = spmin(fun_flat, x0, jac=jac, args=args, method=method, hess=hess, hessp=hessp, bounds=bounds, 
                  constraints=constraints, tol=tol, callback=callback, options=options)

    if descriptive == True:
        return optim
    else:
        return optim['x']
