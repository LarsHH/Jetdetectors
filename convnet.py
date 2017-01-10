import theano
from theano import tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from load_higgs_hdf5 import load_data
from nnet_tools import dropout, sigmoid, tanh, floatX, init_weights, init_biases, RMSprop
from load_higgs import higgs_pkl
import pyroc
import cPickle
import os



srng = RandomStreams()

def model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = tanh(T.dot(X, w_h) + b_h)

    h = dropout(h, p_drop_hidden[0])
    h2 = tanh(T.dot(h, w_h2) + b_h2)

    h2 = dropout(h2, p_drop_hidden[1])
    h3 = tanh(T.dot(h2, w_h3) + b_h3)
    
    h3 = dropout(h3, p_drop_hidden[2])
    h4 = tanh(T.dot(h3, w_h4) + b_h4)

    h4 = dropout(h4, p_drop_hidden[3])
    py_x = sigmoid(T.dot(h4, w_o) + b_o)
    return h, h2, h3, h4, py_x

trX, trY, vX, vY, teX, teY = higgs_pkl()

trX = trX[0:500000]
trY = trY[0:500000]
vX = vX[0:10000]
vY = vY[0:10000]

X = T.fmatrix()
Y = T.fmatrix()

import theano
from theano import tensor as T
import numpy as np
from theano.tensor.nnet.conv import conv2d
X = T.ftensor4()
w = theano.shared(value=np.zeros((10, 1, 4, 4), dtype='float32'), name='w', borrow=True)
y = conv2d(X, w, border_mode='valid', subsample=(1, 4))
f = theano.function(inputs=[X], outputs=y, allow_input_downcast=True)

x = np.zeros((112,))
x[48:64] = np.arange(16)
x = x.reshape(-1, 1, 7, 16)
f(x).shape





jets = T.fmatrix('Jets')
lepton_missing_E = T.fmatrix('lepton_and_missing_energy')
high_level = T.fmatrix('high_level')









n_input_units = 28
n_hidden_units = 300
n_output_units = 1

# Initialize Weights
w_h = init_weights((n_input_units, n_hidden_units), name='weights_layer_1', scale=0.1)
b_h = init_biases(n_hidden_units, name='bias_layer_1')
w_h2 = init_weights((n_hidden_units, n_hidden_units), name='weights_layer_2')
b_h2 = init_biases(n_hidden_units, name='bias_layer_2')
w_h3 = init_weights((n_hidden_units, n_hidden_units), name='weights_layer_3')
b_h3 = init_biases(n_hidden_units, name='bias_layer_3')
w_h4 = init_weights((n_hidden_units, n_hidden_units), name='weights_layer_4')
b_h4 = init_biases(n_hidden_units, name='bias_layer_4')
w_o = init_weights((n_hidden_units, n_output_units), name='weights_layer_y', scale=0.001)
b_o = init_biases(n_output_units, name='bias_layer_y')


noise_h, noise_h2, noise_h3, noise_h4, noise_py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, 0.8, [0.5,0.5,0.5,0.5])

h, h2, h3, h4, py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, 0., [0.,0.,0.,0.])
y_x = T.round(py_x, mode="half_away_from_zero")

L2reg = .00001
cost = T.mean(T.nnet.binary_crossentropy(noise_py_x, Y)) + L2reg * ((w_h ** 2).sum() + (w_h2 ** 2).sum())
params = [w_h, w_h2, w_h3, w_o, b_h, b_h2, b_h3, b_o]
current_epoch = theano.shared(floatX(0.))

### Learning rate
learning_rate = 0.1

### Momentum
end_momentum = 200
start_momentum = 0

updates = RMSprop(cost, params, lr=learning_rate)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[py_x, y_x], allow_input_downcast=True)

# Define min validation error and max validation AUC
min_error = 0.99
max_AUC = 0.01
best_epoch = 0
batches_seen = 0



for i in range(1000):
    for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
        cost = train(trX[start:end], trY[start:end])
        batches_seen += 1
    pred = predict(vX)
    current_epoch.set_value(floatX(i))
    this_error = 1-np.mean(vY == pred[1])
    if this_error < min_error:
        min_error = this_error
    this_AUC = pyroc.ROCData(zip(vY, pred[0])).auc()
    if this_AUC > max_AUC:
        max_AUC = this_AUC
        best_epoch = i
        f = file(os.path.basename(__file__)+'_best.pkl', 'wb')
        for p in params:
           cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    if i > end_momentum and (i-best_epoch)>10:
        break


    print "Epoch {0} \t Error={1}".format(i, this_error)
    print "          \t AUC={0}".format(this_AUC)
    if not(i%5):
        print "          \t \t min Error={0} \t max AUC={1}".format(min_error, max_AUC)


print "---Training ended---"
f = file(os.path.basename(__file__)+'_best.pkl', 'rb')
for p in params:
   p.set_value(cPickle.load(f))
f.close()

pred = predict(teX)
test_error = 1-np.mean(teY == pred[1])
test_AUC = pyroc.ROCData(zip(teY, pred[0])).auc()
print "Test \t Error={0}".format(test_error)
print "     \t AUC={0}".format(test_AUC)
