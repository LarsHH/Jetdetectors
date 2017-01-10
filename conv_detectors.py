import theano
import theano.tensor as T
from nnet_tools import init_weights, tanh, rectify
import numpy as np
from theano.tensor.nnet.conv import conv2d

def pad(X, axis=1):
    return T.concatenate([T.zeros_like(X), T.zeros_like(X), T.zeros_like(X),
                          X, T.zeros_like(X), T.zeros_like(X), T.zeros_like(X)], axis=axis)


def conv_detector(X):
    n_detectors_layer_1 = 20
    n_detectors_layer_2 = 30

    n = X.shape[0]

    input = pad(X)

    w1 = init_weights((n_detectors_layer_1, 1, 4, 4), name="weights_layer_1", scale=0.1)
    w2 = init_weights((n_detectors_layer_2, n_detectors_layer_1, 4, 4), name="weights_layer_2")

    h1 = rectify(conv2d(input.reshape((n, 1, 7, 16)), w1, border_mode='valid', subsample=(1, 4)))

    input_2 = h1.reshape((n, n_detectors_layer_1, 1, 16))

    h2 = rectify(conv2d(pad(input_2, axis=3).reshape((n, n_detectors_layer_1, 7, 16)), w2, border_mode='valid', subsample=(1, 4)))

    output = h2.reshape((n, 4*4*n_detectors_layer_2))
    return output, [w1, w2]

if __name__=="__main__":
    X = T.fmatrix('X')
    f = theano.function(inputs=[X], outputs=conv_detector(X)[0], allow_input_downcast=True)

    x = np.ones((2, 16))
    print f(x)
    print f(x).shape