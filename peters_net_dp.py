import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
# from load_higgs_hdf5 import load_data
from nnet_tools import dropout, sigmoid, tanh, floatX, init_weights, momentum
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

# trX, trY = load_data()
trX = trX[0:500000]
trY = trY[0:500000]
# vX, vY = load_data('valid')
vX = vX[0:10000]
vY = vY[0:10000]

X = T.fmatrix()
Y = T.fmatrix()

w_h = theano.shared(floatX(np.random.randn(*(28, 300)) * 0.1))
b_h = theano.shared(value=np.zeros((300,), dtype=theano.config.floatX), name='b_h', borrow=True)
w_h2 = init_weights((300, 300))
b_h2 = theano.shared(value=np.zeros((300,), dtype=theano.config.floatX), name='b_h2', borrow=True)
w_h3 = init_weights((300, 300))
b_h3 = theano.shared(value=np.zeros((300,), dtype=theano.config.floatX), name='b_h3', borrow=True)
w_h4 = init_weights((300, 300))
b_h4 = theano.shared(value=np.zeros((300,), dtype=theano.config.floatX), name='b_h4', borrow=True)
w_o = theano.shared(floatX(np.random.randn(*(300, 1)) * 0.001))
b_o = theano.shared(value=np.zeros((1,), dtype=theano.config.floatX), name='b_o', borrow=True)


noise_h, noise_h2, noise_h3, noise_h4, noise_py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, 0.8, [0.5,0.5,0.5,0.5])

h, h2, h3, h4, py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, 0., [0.,0.,0.,0.])
y_x = T.round(py_x, mode="half_away_from_zero")

L2reg = .00001
cost = T.mean(T.nnet.binary_crossentropy(noise_py_x, Y)) + L2reg * ((w_h ** 2).sum() + (w_h2 ** 2).sum())
params = [w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o]
current_epoch = theano.shared(floatX(0.))

### Learning rate
init_lr = 0.01
decay_factor = 1.0000002
min_lr = 0.000001
lr = theano.shared(floatX(init_lr))
batches_seen = 0.

### Momentum
end_momentum = 200
start_momentum = 0

updates = momentum(cost, params, current_epoch, lr, init_momentum=0.09)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[py_x, y_x], allow_input_downcast=True)

# Define min validation error and max validation AUC
min_error = 0.99
max_AUC = 0.01
best_epoch = 0



for i in range(1000):
    for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
        cost = train(trX[start:end], trY[start:end])
        batches_seen += 1
        # Learning rate decay
        lr.set_value(floatX(np.amax((init_lr / (decay_factor ** batches_seen), min_lr))))
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
    print "          \t Learning rate={0}".format(lr.get_value())
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
