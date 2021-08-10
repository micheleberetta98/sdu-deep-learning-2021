import numpy as np

a = np.full((2, 3), 4)
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.eye(2, 3)
d = a + b + c

# print('a =\n', a)
# print('b =\n', b)
# print('c =\n', c)
# print('d =\n', d)

a = np.array([[1, 2, 3, 4, 5],
              [5, 4, 3, 2, 1],
              [6, 7, 8, 9, 0],
              [0, 9, 8, 7, 6]])

# Get the third row from 'a' as a rank 2 array
print(a[2])    # This is rank 1 [6, 7, 8, 9, 0]
print(a[2:3])  # This is rank 2 [[6, 7, 8, 9, 0]]

# Sum the rows of 'a'
print(np.sum(a, 1))

# Get the transpose of 'a'
print(np.transpose(a))
