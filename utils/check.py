import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import skip_init
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
    from .layers import *
    from .layer_utils import *
    from .optim import *
    from .gradient_check import *
    from .monitor import *
except ImportError:
    print("Relative import failed. Trying absolute import.")
    from layers import *
    from layer_utils import *
    from optim import *
    from gradient_check import *
    from monitor import *

class tinyNeuralNetwork(nn.Module):
    def __init__(self, seed=42, weight_scale=1e-2) -> None:
        super().__init__()
        # To use numpy initialization, we need to skip the initialization of the weights and biases
        # see details: https://pytorch.org/tutorials/prototype/skip_param_init.html
        self.fc1 = skip_init(nn.Linear, 100, 1000)
        self.fc2 = skip_init(nn.Linear, 1000, 10)
        self.norm = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(p=0.5)

        # initialize weights and biases with numpy random normal
        if seed:
            np.random.seed(seed=seed)
        # Linear module details: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # note that y = xA^T + b, so the shape of weight is (out_features, in_features)
        self.fc1.weight.data = torch.from_numpy(weight_scale * np.random.normal(loc=0.0, scale=weight_scale, size=(100, 1000)).T)
        self.fc1.bias.data = torch.from_numpy(np.zeros(1000))
        self.fc2.weight.data = torch.from_numpy(weight_scale * np.random.normal(loc=0.0, scale=weight_scale, size=(1000, 10)).T)
        self.fc2.bias.data = torch.from_numpy(np.zeros(10))
        # print(f"fc1 weight {self.fc1.weight.data.T}")
        # print(f"fc2 weight {self.fc2.weight.data.T}")   

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc1(x)
        y = self.norm(y)
        y = self.relu(y)
        # y = self.dropout(y)
        y = self.fc2(y)
        loss = self.softmax(y)
        return y, loss

class manualNN():
    def __init__(
        self, 
        input_dim, 
        hidden_dims, 
        num_classes=10, 
        dropout_keep_ratio = 1.0, 
        normalization = None,
        weight_scale=1e-2, 
        learning_rate=1e-3,
        reg=0.0,
        seed=None,
        dtype=np.float32
    ) -> None:
        self.num_layers = len(hidden_dims) + 1
        self.use_dropout = (dropout_keep_ratio != 1.0)
        self.normalization = normalization
        self.reg = reg
        self.params = {}
        """
        Network Structure(normalization, dropout, regularization are not implemented):
        {affine -> relu} * n -> affine -> softmax
        Next step:
        {affine -> batchnorm/layernorm -> relu -> [dropout]} * n -> affine -> softmax

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
        if self.normalization in ["batchnorm", "layernorm"]:
            for i in range(self.num_layers - 1):
                self.params[f"gamma{i+1}"] = np.ones(hidden_dims[i])
                self.params[f"beta{i+1}"] = np.zeros(hidden_dims[i])
        elif self.normalization:
            print(f"What do you expect from me? I don't know what {self.normalization} is.")

        # Initialize dropout parameters
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed:
                self.dropout_param["seed"] = seed


        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        # Initialize config parameters for each module and store them in self.optim_config
        self.optim_config = {}
        self.optim_configs = {}
        self.optim_config["learning_rate"] = learning_rate
        for p in self.params:
            self.optim_configs[p] = {}
            for k, v in self.optim_config.items():
                self.optim_configs[p][k] = v

        # store loss and accuracy history
        self.loss_hist = []
        self.acc_hist = []
    

    def train(self, X, y, optim : str, batch_size=None, epochs=100, lr=1e-3, print_every=10, verbose=False):

        optim_dict = {"sgd":sgd, "sgd_momentum":sgd_momentum, "rmsprop":rmsprop, "adam":adam}
        optim = optim_dict[optim]
        # only learning rate is needed for now, but learning rate decay will be added later
        self.optim_config["learning_rate"] = lr
        for p in self.params:
            self.optim_configs[p] = {}
            for k, v in self.optim_config.items():
                self.optim_configs[p][k] = v
        
        if batch_size is None:
            batch_size = X.shape[0]
        
        for i in range(epochs):
            batch_mask = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[batch_mask], y[batch_mask]
            pred_y, caches = self.predict(X_batch, y_batch)
            loss, dout = softmax_loss(pred_y, y_batch)
            self.loss_hist.append(loss)
            acc = np.mean(np.argmax(pred_y, axis=1) == y_batch)
            self.acc_hist.append(acc)
            self.backward(dout, caches, optim, lr)
            if i % print_every == 0:
                print(f"Epoch {i}: loss {loss} accuracy {acc}")
                if verbose:
                    get_memory(verbose=True)
        

    @tic_toc
    def predict(self, X, y=None):
        """
        This function implement a forward pass of the network.
        If you pass y, it will trate it as a training process.
        If you don't pass y, it will trate it as a testing process.
        """
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        caches = []
        out = X
        for i in range(1, self.num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            if self.normalization == None:
                out, cache = affine_relu_forward(out, W, b)
                caches.append(cache)

            elif self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                bn_param = self.bn_params[i - 1]
                out, cache = affine_bn_relu_forward(out, W, b, gamma, beta, bn_param)
                caches.append(cache)

            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                ln_param = self.bn_params[i - 1]
                out, cache = affine_ln_relu_forward(out, W, b, gamma, beta, ln_param)
                caches.append(cache)

            if self.use_dropout:
                out, cache = dropout_forward(out, self.dropout_param)
                caches.append(cache)

        W = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]
        out, cache = affine_forward(out, W, b)
        caches.append(cache)
        scores = out
        get_memory(verbose=True)

        return scores, caches

    @tic_toc
    def backward(self, dout, caches, optim, lr):
        reg, grads = self.reg, {}
        
        # Backward pass: compute gradients
        dout, dw, db = affine_backward(dout, caches[-1])
        grads[f"dW{self.num_layers}"] = dw
        grads[f"db{self.num_layers}"] = db


        for i in range(self.num_layers - 2, -1, -1):
            if self.use_dropout:
                cache = caches[i]
                dout = dropout_backward(dout, cache)
            if self.normalization == None:
                cache = caches[i]
                dout, dw, db = affine_relu_backward(dout, cache)
                dw += reg * self.params[f"W{i+1}"]
                grads[f"dW{i+1}"] = dw
                grads[f"db{i+1}"] = db
            elif self.normalization == "batchnorm":
                cache = caches[i]
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache)
                dw += reg * self.params[f"W{i+1}"]
                grads[f"dW{i+1}"] = dw
                grads[f"db{i+1}"] = db
                grads[f"dgamma{i+1}"] = dgamma
                grads[f"dbeta{i+1}"] = dbeta
            elif self.normalization == "layernorm":
                cache = caches[i]
                dout, dw, db, dgamma, dbeta = affine_ln_relu_backward(dout, cache)
                dw += reg * self.params[f"W{i+1}"]
                grads[f"dW{i+1}"] = dw
                grads[f"db{i+1}"] = db
                grads[f"dgamma{i+1}"] = dgamma
                grads[f"dbeta{i+1}"] = dbeta

        # Update parameters
        for p, w in self.params.items():
            dw = grads[f"d{p}"]
            config = self.optim_configs[p]
            next_w, next_config = optim(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config

        return grads

    def save_model(self, path="model.json", epoch=None):
        save_path = f"epoch_{epoch}_{path}"
        with open(save_path, "w") as f:
            json.dump(self.params, f)

    def load_model(self, path="model.json"):
        if path.startswith("epoch_"):
            epoch = int(path.split("_")[1])
            print(f"Loading model which has been trained {epoch} epoches...")
        with open(path, "r") as f:
            self.params = json.load(f)

    def get_param(self):
        return self.params




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

    def test_forward():
        model = tinyNeuralNetwork()
        model.to(device).float()
        model.train()

        test_model = manualNN(100, [1000], 10, seed=42, normalization="batchnorm", 
                        dropout_keep_ratio=1.0, weight_scale=1e-2)

        # Generate random input by numpy
        np.random.seed(42)
        x = np.random.normal(loc=0.0, scale=100, size=(20000, 100))
        y = np.random.randint(0, 10, size=(20000,10))
        torch_input = torch.from_numpy(x).to(device).float()
        torch_out, _ = model(torch_input)
        torch_out_np = torch_out.cpu().detach().numpy()

        out, _ = test_model.predict(x, y)

        # Check the correctness of the forward pass
        err = rel_error(torch_out_np, out)
        print('\n'*2 + "Check the correctness of the forward pass")
        if err < 1e-10:
            print(f"Forward pass is correct. Rel Error: {err}")
        else:
            print(f"Forward pass is wrong. Rel Error: {err}")

    def test_backward():     
        model = tinyNeuralNetwork(weight_scale=1)
        model.to(device).float()
        model.train()
        creterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        test_model = manualNN(100, [1000], 10, seed=42, normalization="batchnorm", 
                        dropout_keep_ratio=1.0, weight_scale=1)

        # Generate random input by numpy
        np.random.seed(42)
        x = np.random.normal(loc=0.0, scale=100, size=(20000, 100))
        y = np.random.randint(0, 10, size=(20000))
        torch_input = torch.from_numpy(x).to(device).float()
        torch_out, _ = model(torch_input)
        torch_y = torch.from_numpy(y).to(device).long()
        torch_loss = creterion(torch_out, torch_y)
        torch_loss.backward()
        optimizer.step()
        updated_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(name, param.grad.shape)
                updated_params.append(param.data.cpu().detach().numpy())

        
        out, caches = test_model.predict(x, y)
        loss, dout = softmax_loss(out, y)
        test_model.backward(dout, caches, optim=adam , lr=1e-3)
        updated_params_manual = test_model.get_param()
        
        for name, param in test_model.get_param().items():
            print(name, param.shape)

        # Check the correctness of the backward pass
        print('\n'*2 + "Check the correctness of the backward pass")
        errs = []
        for value1, value2 in zip(updated_params, updated_params_manual.values()):
            try:
                err = rel_error(value1, value2)
            except:
                err = rel_error(value1.T, value2)
            errs.append(err)
        print(f"errs: {errs}")        
   
        
    test_forward()
    test_backward()