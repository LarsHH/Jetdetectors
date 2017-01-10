import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
# from load_higgs_hdf5 import load_data
from load_higgs import higgs_pkl
import pyroc
import cPickle
import os


srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.05))


def init_bias(dim):
    return theano.shared(value=np.zeros((dim,), dtype=theano.config.floatX), borrow=True)


def sigmoid(X):
    return T.nnet.sigmoid(X)


def tanh(X):
    return T.tanh(X)


def momentum(cost, params, current_epoch, lr, init_momentum):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        vel = theano.shared(p.get_value() * 0.)
        current_momentum = theano.shared(floatX(init_momentum))
        vel_new = current_momentum * vel - lr * g
        momentum_new = T.le(current_epoch, 200.) * (current_epoch * (0.99 - 0.09) / 200. + 0.09) + T.gt(current_epoch,
                                                                                                        200.) * 0.99
        updates.append((vel, vel_new))
        updates.append((p, p + vel_new))
        updates.append((current_momentum, momentum_new))
    return updates


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, gamma, beta, gamma2, beta2, gamma3, beta3,
          gamma4, beta4, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = tanh(T.dot(X, w_h) + b_h)

    h = dropout(h, p_drop_hidden[0])
    h = gamma * (h - T.mean(h, axis=0)) / T.sqrt(T.var(h, axis=0)) + beta
    h2 = tanh(T.dot(h, w_h2) + b_h2)

    h2 = dropout(h2, p_drop_hidden[1])
    h2 = gamma2 * (h2 - T.mean(h2, axis=0)) / T.sqrt(T.var(h2, axis=0)) + beta2
    h3 = tanh(T.dot(h2, w_h3) + b_h3)

    h3 = dropout(h3, p_drop_hidden[2])
    h3 = gamma3 * (h3 - T.mean(h3, axis=0)) / T.sqrt(T.var(h3, axis=0)) + beta3
    h4 = tanh(T.dot(h3, w_h4) + b_h4)

    h4 = dropout(h4, p_drop_hidden[3])
    h4 = gamma4 * (h4 - T.mean(h4, axis=0)) / T.sqrt(T.var(h4, axis=0)) + beta4
    py_x = sigmoid(T.dot(h4, w_o) + b_o)
    return h, h2, h3, h4, py_x


def companions(wc_h, wc_h2, wc_h3, h, h2, h3, bc_h, bc_h2, bc_h3):
    c = sigmoid(T.dot(h, wc_h) + bc_h)

    c2 = sigmoid(T.dot(h2, wc_h2) + bc_h2)

    c3 = sigmoid(T.dot(h3, wc_h3) + bc_h3)
    return c, c2, c3


trX, trY, vX, vY, teX, teY = higgs_pkl()

# trX, trY = load_data()
# trX = trX[0:500000]
# trY = trY[0:500000]
# vX, vY = load_data('valid')
# vX = vX[0:10000]
# vY = vY[0:10000]
#
X = T.fmatrix()
Y = T.fmatrix()


# Network parameters
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

# Companion weights
wc_h = theano.shared(floatX(np.random.randn(*(300, 1)) * 0.001))
bc_h = init_bias(1)
wc_h2 = theano.shared(floatX(np.random.randn(*(300, 1)) * 0.001))
bc_h2 = init_bias(1)
wc_h3 = theano.shared(floatX(np.random.randn(*(300, 1)) * 0.001))
bc_h3 = init_bias(1)

# Batch norm parameters
gamma = init_weights((300,))
beta = init_weights((300,))
gamma2 = init_weights((300,))
beta2 = init_weights((300,))
gamma3 = init_weights((300,))
beta3 = init_weights((300,))
gamma4 = init_weights((300,))
beta4 = init_weights((300,))

# Momentum
end_momentum = 200
start_momentum = 0

# Miscellaneous
init_lr = 0.1
decay_factor = 1.0000002
min_lr = 0.000001
lr = theano.shared(floatX(init_lr))
batches_seen = 0.
current_epoch = theano.shared(floatX(0.))
L2reg = .00001
c_reg = T.maximum(.3 * (1 - current_epoch/end_momentum), 0.)


# Models
noise_h, noise_h2, noise_h3, noise_h4, noise_py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o,
                                                          gamma, beta, gamma2, beta2, gamma3, beta3, gamma4, beta4, 0.,
                                                          [0., 0., 0., 0.5])
c, c2, c3 = companions(wc_h, wc_h2, wc_h3, noise_h, noise_h2, noise_h3, bc_h, bc_h2, bc_h3)
h, h2, h3, h4, py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o,
                            gamma, beta, gamma2, beta2, gamma3, beta3, gamma4, beta4, 0., [0., 0., 0., 0.])
y_x = T.round(py_x, mode="half_away_from_zero")


def cross_entropy(yhat, y):
    return T.mean(T.nnet.binary_crossentropy(yhat, y))


def L2(L2_reg, params):
    l2_cost = theano.shared(0.)
    for p in params:
        l2_cost += (p ** 2).sum()
    return L2_reg * l2_cost


# Training
cost = cross_entropy(noise_py_x, Y) + c_reg * cross_entropy(c, Y) + c_reg * cross_entropy(c2, Y) \
    + c_reg * cross_entropy(c3, Y) + L2(L2reg, [w_h, w_h2, w_h3, w_h4, wc_h, wc_h2, wc_h3])
params = [w_h, w_h2, w_h3, w_h4, w_o, b_h, b_h2, b_h3, b_h4, b_o, wc_h, wc_h2, wc_h3, bc_h, bc_h2, bc_h3,
          gamma, beta, gamma2, beta2, gamma3, beta3, gamma4, beta4]
updates = momentum(cost, params, current_epoch, lr, init_momentum=0.09)

# Compiling
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
    this_error = 1 - np.mean(vY == pred[1])
    if this_error < min_error:
        min_error = this_error
    this_AUC = pyroc.ROCData(zip(vY, pred[0])).auc()
    if this_AUC > max_AUC:
        max_AUC = this_AUC
        best_epoch = i
        f = file(os.path.basename(__file__) + '_best.pkl', 'wb')
        for p in params:
            cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    if i > end_momentum and (i - best_epoch) > 10:
        break

    print "Epoch {0} \t Error={1}".format(i, this_error)
    print "          \t AUC={0}".format(this_AUC)
    if not (i % 5):
        print "          \t \t min Error={0} \t max AUC={1}".format(min_error, max_AUC)

print "---Training ended---"
f = file(os.path.basename(__file__) + '_best.pkl', 'rb')
for p in params:
    p.set_value(cPickle.load(f))
f.close()

pred = predict(teX)
test_error = 1 - np.mean(teY == pred[1])
test_AUC = pyroc.ROCData(zip(teY, pred[0])).auc()
print "Test \t Error={0}".format(test_error)
print "     \t AUC={0}".format(test_AUC)
