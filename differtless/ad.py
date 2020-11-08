"""

Auto-Differentiation Module:

Contents:
    - Preprocessing function
    - FuncInput class
    - forward AD function
    - Minimize function

"""

"""

FuncInput class.

Takes a value and a seed (list of derivatives). Fundmanetal operations (addition, multiplication, etc.)
are overwritten so that they will return a FuncInput object with updated value and derivative values.

TODO: Write actual DocString

"""

class FuncInput():

    def __init__(self, value, seed):
        self.val_ = value
        self.ders_ = seed

    def __str__(self):
        return f'FuncInput object with value {self.val_} and gradients {self.ders_} with respect to each input'

    def __repr__(self):
        return f'FuncInput({sel.val_}, {sel.ders_})'
        
