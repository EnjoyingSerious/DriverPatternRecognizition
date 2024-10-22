import matplotlib.pyplot as plt
import numpy as np

shape = (2, 1, 1, 1)
four_dim_array = np.random.rand(*shape)
print(four_dim_array, type(four_dim_array), four_dim_array.shape)

four_dim_array = [pc for pc in four_dim_array]
print(four_dim_array, type(four_dim_array), len(four_dim_array))