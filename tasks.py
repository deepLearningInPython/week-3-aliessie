import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    return input_length - kernel_length + 1

# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))
# Intuition: the 1 accounts for the initial position. For the rest of the kernel we can hover the kernel to the right the number of times the difference of the lenghts allows. 


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    input_len = len(input_array)
    kernel_len = len(kernel_array)
    output_len = input_len - kernel_len + 1

    output = np.zeros(output_len)

    for i in range(output_len):
        output[i] = np.sum(input_array[i:i+kernel_len]*kernel_array)
    return output

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1 
    return(output_height, output_width)


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel_matrix.shape
    
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            window = input_matrix[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = (window * kernel_matrix).sum()
    
    return output
# -----------------------------------------------