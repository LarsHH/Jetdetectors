import theano
import theano.tensor as T
from nnet_tools import init_weights, tanh, rectify
import numpy as np

def jet_detector(X):

    n_detectors_layer_1 = 5
    n_jets_layer_1 = 4
    n_detectors_layer_2 = 5
    n_jets_layer_2 = 4

    # Takes 16 dim vector, applies W to it, returns resultant vector
    def apply_detector(W, jet, n_jets):
        map_i = []
        for start, end in zip(range(0, 16, n_jets), range(4, 16+4, n_jets)):
            map_i.append(rectify(T.dot(W, jet[start:end])))
        return T.concatenate(map_i, axis=0)

    detectors_layer_1 = []
    # Initialize weights
    for i in range(n_detectors_layer_1):
        detectors_layer_1.append(init_weights((4, 4), name="weights_layer_1_{0}".format(i)), scale=0.1)

    maps_layer_1 = []
    for W in detectors_layer_1:
        map_i = apply_detector(W, X.T, n_jets_layer_1)
        maps_layer_1.append(map_i)

    detectors_layer_2 = []
    # Initialize weights
    for i in range(n_detectors_layer_2):
        detectors_layer_2.append(init_weights((4, 4), name="weights_layer_2_{0}".format(i)))

    maps_layer_2 = []
    for map in maps_layer_1:
        for W in detectors_layer_2:
            map_i = apply_detector(W, map, n_jets_layer_2)
            maps_layer_2.append(map_i)

    jet_detector_output = T.concatenate(maps_layer_2, axis=0)

    params = detectors_layer_1 + detectors_layer_2

    return jet_detector_output.T, params


if __name__=="__main__":
X = T.fmatrix('X')
output = jet_detector(X)
#grads = T.grad(cost=T.sum(output[0]), wrt=output[1])
f = theano.function(inputs=[X], outputs=output[0], allow_input_downcast=True)
#g = theano.function(inputs=[X], outputs=grads, allow_input_downcast=True)

x = np.ones((2, 16))
print f(x)
print f(x).shape

from load_higgs import higgs_pkl
trX, trY, vX, vY, teX, teY = higgs_pkl()
print f(trX[:128, 6:22]).shape
