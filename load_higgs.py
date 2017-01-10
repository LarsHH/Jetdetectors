import numpy as np
import csv
import gzip
import cPickle

def higgs(n=11000000):
    print "Loading data..."
    data = np.zeros((n,29))
    path = '../../IGB/HIGGS/HIGGS.csv.gz'
    with gzip.open(path) as f:
        reader = csv.reader(f)
        for i in range(n):
            data[i, :] = reader.next()

    X = data[:, 1:]
    y = np.atleast_2d(data[:, 0]).T
    n_train = 500000
    n_valid = 25000
    print "Loading data complete."
    return X[0:n_train, :], y[0:n_train], X[n_train:(n_train+n_valid), :], y[n_train:(n_train+n_valid)], \
           X[(n_train+n_valid):(n_train+2*n_valid), :], y[(n_train+n_valid):(n_train+2*n_valid)]

def higgs_pkl():
    print "Loading data from pickle."
    path = '../data/HIGGS/'

    f = file(path+'trX.pkl', 'rb')
    trX = cPickle.load(f)
    f.close()

    f = file(path+'trY.pkl', 'rb')
    trY = cPickle.load(f)
    f.close()

    f = file(path+'teX.pkl', 'rb')
    teX = cPickle.load(f)
    f.close()

    f = file(path+'teY.pkl', 'rb')
    teY = cPickle.load(f)
    f.close()

    f = file(path+'vX.pkl', 'rb')
    vX = cPickle.load(f)
    f.close()

    f = file(path+'vY.pkl', 'rb')
    vY = cPickle.load(f)
    f.close()
    print "Loading data complete."
    return trX, trY, vX, vY, teX, teY

def higgs_pkl_inter():
    print "Loading data from pickle."
    path = '../data/HIGGSinter/'

    f = file(path+'trX_inter.pkl', 'rb')
    trX = cPickle.load(f)
    f.close()

    f = file(path+'trY.pkl', 'rb')
    trY = cPickle.load(f)
    f.close()

    f = file(path+'teX_inter.pkl', 'rb')
    teX = cPickle.load(f)
    f.close()

    f = file(path+'teY.pkl', 'rb')
    teY = cPickle.load(f)
    f.close()

    f = file(path+'vX_inter.pkl', 'rb')
    vX = cPickle.load(f)
    f.close()

    f = file(path+'vY.pkl', 'rb')
    vY = cPickle.load(f)
    f.close()
    print "Loading data complete."
    return trX, trY, vX, vY, teX, teY
