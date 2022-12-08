import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import skip_init
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
    from .layers import *
    from .gradient_check import *
    from .monitor import *
except:
    print("Relative import failed. Trying absolute import.")
    from layers import *
    from gradient_check import *
    from monitor import *

class tinyNeuralNetwork(nn.Module):
    def __init__(self, seed=42, weight_scale=1e-2, device=None) -> None:
        super().__init__()
        # To use numpy initialization, we need to skip the initialization of the weights and biases
        # see details: https://pytorch.org/tutorials/prototype/skip_param_init.html
        self.fc1 = skip_init(nn.Linear, 100, 1000)
        self.fc2 = skip_init(nn.Linear, 1000, 10)

        # initialize weights and biases with numpy random normal
        if seed:
            np.random.seed(seed=seed)
        # Linear module details: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # note that y = xA^T + b, so the shape of weight is (out_features, in_features)
        self.fc1.weight.data = torch.from_numpy(weight_scale * np.random.normal(loc=0.0, scale=weight_scale, size=(100, 1000)).T).double()
        self.fc1.bias.data = torch.from_numpy(np.zeros(1000)).double()
        self.fc2.weight.data = torch.from_numpy(weight_scale * np.random.normal(loc=0.0, scale=weight_scale, size=(1000, 10)).T).double()
        self.fc2.bias.data = torch.from_numpy(np.zeros(10)).double()
        # print(f"fc1 weight {self.fc1.weight.data.T}")
        # print(f"fc2 weight {self.fc2.weight.data.T}")
        if device:
            self.fc1.to(device)
            self.fc2.to(device)     

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        loss = self.softmax(y)
        return y, loss

class manualNN():
    def __init__(
        self, 
        input_dim, 
        hidden_dims, 
        num_classes=10, 
        dropout_ratio = 0.0, 
        normalization = None,
        weight_scale=1e-2, 
        reg=0.0,
        seed=None,
        dtype=np.float64
    ) -> None:
        self.num_layers = len(hidden_dims) + 1
        self.dropout = (dropout_ratio != 0.0)
        self.normalization = normalization
        self.reg = reg
        self.params = {}
        """
        Network Structure(normalization, dropout, regularization are not implemented):
        [affine -> relu] * n -> affine -> softmax
        hidden_dims: list of integers giving the size of each hidden layer.
        """

        if seed:
            np.random.seed(seed)
        
        # Initialize weights and biases, store them in self.params
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(layer_dims) - 1):
            layer_input_dim = layer_dims[i]
            layer_output_dim = layer_dims[i + 1]
            self.params[f"W{i+1}"] = weight_scale * np.random.normal(loc=0.0, scale=weight_scale
                , size=(layer_input_dim, layer_output_dim))
            self.params[f"b{i+1}"] = np.zeros(layer_output_dim)
            # print(f"W{i+1}: {self.params[f'W{i+1}']}")

        # Initialize normalization parameters
        if self.normalization:
            pass

        # Initialize dropout parameters
        if self.dropout:
            pass

        # Initialize regularization parameters
        if self.reg:
            pass
    
    @tic_toc
    def train(self, X, y):
        pred_y = self.test(X)
        loss = self.loss(pred_y, y)
        self.backward(loss)
        pass

    @tic_toc
    def test(self, X):
        """
        Normalization, dropout, regularization are not implemented!
        """
        caches = []
        out = X
        for i in range(1, self.num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            out, cache = affine_forward(out, W, b)
            caches.append(cache)
            out, cache = relu_forward(out)
            caches.append(cache)
        W = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]
        out, cache = affine_forward(out, W, b)
        caches.append(cache)
        scores = out
        get_memory(verbose=True)
        return scores

    def loss(self, X, y):
        pass

    def backward(self, loss):
        pass


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':
    """
    1. Use torch autograd to check the correctness of the backward pass of our naive implementation.
    2. Current implementation of conv_forward and conv_backward is very slow. Check the run-time to see
       whether it can be accepted.
    3. Check the memory usage of the forward and backward pass.
    """
    get_memory(verbose=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = tinyNeuralNetwork(device=device)
    model.to(device)

    test_model = manualNN(100, [1000], 10, seed=42)

    # Generate random input by numpy
    np.random.seed(42)
    x = np.random.normal(loc=0.0, scale=1e-2, size=(2, 100))
    torch_input = torch.from_numpy(x).double().to(device)
    torch_out, _ = model(torch_input)
    torch_out_np = torch_out.cpu().detach().numpy()

    out = test_model.test(x)

    # Check the correctness of the forward pass
    err = rel_error(torch_out_np, out)
    print('\n'*4 + "Check the correctness of the forward pass")
    if err < 1e-10:
        print(f"Forward pass is correct. Rel Error: {err}")
    else:
        print(f"Forward pass is wrong. Rel Error: {err}")
    