from .layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Forward pass for the affine-bn-relu convenience layer.
  """
  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine_bn_relu convenience layer.
  """
  fc_cache, bn_cache, relu_cache = cache
  dan = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward(dan, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta 


def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
  a, fc_cache = affine_forward(x, w, b)
  an, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
  out, relu_cache = relu_forward(an)
  cache = (fc_cache, ln_cache, relu_cache)
  return out, cache

def affine_ln_relu_backward(dout, cache):
  """
  Backward pass for the affine_bn_relu convenience layer.
  """
  fc_cache, ln_cache, relu_cache = cache
  dan = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = layernorm_backward(dan, ln_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta 

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
