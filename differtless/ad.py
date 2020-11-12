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


