import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from gradient_check import eval_numerical_gradient_array

from watch_memory import get_memory

try:
    from .layers import *
except:
    print("Relative import failed. Trying absolute import.")
    from layers import *

if __name__ == '__main__':
    """
    1. Use torch autograd to check the correctness of the backward pass of our naive implementation.
    2. Current implementation of conv_forward and conv_backward is very slow. Check the run-time to see
       whether it can be accepted.
    3. Check the memory usage of the forward and backward pass.
    """
    get_memory()