import numpy as np

# These functions are implented without carefully checked, so maybe you will find some ill results.

def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None


    x_re = x.reshape(x.shape[0], -1)
    out = np.dot(x_re, w) + b


    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None


    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
    db = np.sum(dout, axis=0)


    return dx, dw, db

def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None


    out = x.copy()
    out[x<0] = 0


    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache


    dx = np.zeros_like(x)
    dx[x>0] = 1
    dx = dx * dout


    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None



    num_train = x.shape[0]
    x = x - np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    x_exp = x.copy()   ### 要保留值，请使用深拷贝
    x[range(num_train), y] /= np.sum(x, axis=1)
    # smooth_loss = - np.sum(np.log(x[range(num_train), y] + 1e-10))
    loss = - np.sum(np.log(x[range(num_train), y]))
    # loss = - np.sum(np.log(x[range(num_train), y] + 1e-10))
    loss /= num_train

    dx = x_exp / np.sum(x_exp, axis=1, keepdims=True)
    dx[range(num_train), y] -= 1
    dx /= num_train


    return loss, dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    # momentum = bn_param.get("momentum", 0.1)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":


        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        running_mean = momentum * running_mean + (1-momentum) * sample_mean
        running_var = momentum * running_var + (1-momentum) * sample_var

        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta

        cache = (x, x_norm, sample_mean, sample_var, gamma, beta, eps)


    elif mode == "test":


        out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta


    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None


    num_train = dout.shape[0]

    (x, x_norm, sample_mean, sample_var, gamma, beta, eps) = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = dout * gamma
    dx = dx_norm * (1 / np.sqrt(sample_var+eps)) - dx_norm * (x-sample_mean) / ((sample_var+eps) ** 1.5 )

    dx = (1 / np.sqrt(sample_var + eps)) * (dx_norm - (1 / num_train) * ( np.sum(dx_norm, axis=0) \
      + (1 / (sample_var + eps)) * (x - sample_mean) * np.sum(dx_norm * (x - sample_mean),axis=0)))



    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)


    N, D = x.shape
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = x_norm * gamma + beta
    cache = (x, x_norm, sample_mean, sample_var, gamma, beta, eps)


    return out, cache

def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None


    (x, layer_norm_x, layer_mean, layer_var, gamma, beta, eps) = cache
    N, D = x.shape
    dgamma = np.sum(layer_norm_x * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dbnx = dout * gamma
    dx = (1 / np.sqrt(layer_var + eps)) * (dbnx - (1 / D) * (np.sum(dbnx, axis=1, keepdims=True)
        + (x - layer_mean) / (layer_var + eps) * np.sum(dbnx * (x - layer_mean), axis=1, keepdims=True)))


    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    paper: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":


        mask = (np.random.random(x.shape) < p ) / p
        out = mask * x


    elif mode == "test":


        out = x



    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":


        dx = dout * mask


    elif mode == "test":
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None


    padding = conv_param['pad']
    stride = conv_param['stride']

    x_pad = np.pad(x, ((0,), (0,), (padding,), (padding,)), mode='constant', constant_values=0)

    N = x.shape[0]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    H = 1 + (x.shape[2] + 2 * padding - w.shape[2]) // stride
    W = 1 + (x.shape[3] + 2 * padding - w.shape[3]) // stride
    out = np.zeros((x.shape[0], F, H, W))
    for n in range(N):
      for i in range(F):
        for j in range(H):
          for k in range(W):
            # print(x_pad[:,:,j*stride:j*stride+HH,k*stride:k*stride+WW].shape)
            # print(w[0].shape)
            # print(w[i,:,:,:][np.newaxis,:,:,:].shape)
            out[n:,i,j,k] = np.sum(x_pad[n,:,j*stride:j*stride+HH,k*stride:k*stride+WW] * w[i,:,:,:]) + b[i]



    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None


    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis = (0,2,3))

    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for k in range(F): #compute dw
                dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N): #compute dx_pad
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] *
                                                  (dout[n, :, i, j])[:,None ,None, None]), axis=0)
    dx = dx_pad[:,:,pad:-pad,pad:-pad]




    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, "Illegal Input dimension of H."
    assert (W - pool_width) % stride == 0, "Illegal Input dimension of W."

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))

    for j in range(H_out):
      for k in range(W_out):
        out[:,:,j,k] = np.max(x[:,:,j*stride:j*stride+pool_height, k*stride:k*stride+pool_width], axis=(2,3)) 

    
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    

    x, pool_param = cache
    dx = np.zeros_like(x)

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, "Illegal Input dimension of H."
    assert (W - pool_width) % stride == 0, "Illegal Input dimension of W."

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    for j in range(H_out):
      for k in range(W_out):
        x_mask = x[:,:,j*stride:j*stride+pool_height, k*stride:k*stride+pool_width]
        flags = np.max(x_mask, axis=(2,3), keepdims=True) == x_mask # only the maximum value get a gradient
        dx[:,:,j*stride:j*stride+pool_height, k*stride:k*stride+pool_width] += flags * (dout[:,:,j,k][:,:,None,None]) 
    

    
    return dx

def max_pool1d_naive_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling1d layer.
    In this case, we only implement a simple case, where the stride is L.
    Inputs:
    - x: Input data, of shape (N, C, L)
    - pool_param: dictionary with the following keys:
      - 'kernel_size': Union[int, Tuple[int]], here is L
      - 'stride': The distance between adjacent pooling regions, here is L
    Outputs:
    - out: Output data, of shape (N, C, 1)

    out(i, j, 1) = max_{1}()
    """
    stride = pool_param['stride']
    N, C, L = x.shape
    assert L % stride == 0, "Illegal Input dimension of L."
    out = np.max(x, axis=2, keepdims=True)
    cache = (x, pool_param)
    return out, cache

def max_pool1d_naive_backward(dout, cache):
    """A naive implementation of the backward pass for a max-pooling1d layer.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, 1)
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x, of shape (N, C, L)
    """
    x, pool_param = cache
    dx = np.zeros_like(x)
    stride = pool_param['stride']
    N, C, L = x.shape
    assert L % stride == 0, "Illegal Input dimension of L."
    for i in range(N):
        for j in range(C):
            x_mask = x[i, j, :]
            flags = np.max(x_mask, axis=0, keepdims=True) == x_mask
            dx[i, j, :] += flags * dout[i, j]
    return dx
