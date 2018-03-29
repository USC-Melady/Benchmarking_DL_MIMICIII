
import numpy as np
import theano
import theano.tensor as TT

def make_theano_shared(X, y=None, borrow=True):

    Xsh = theano.shared(np.asarray(X, dtype=theano.config.floatX),
                        borrow=borrow)
    if y is not None:
        if len(y.shape) == 1:
            y = y.reshape([y.shape[0], 1])
        ysh = TT.cast(theano.shared(np.asarray(y, dtype=theano.config.floatX),
                                    borrow=borrow), 'int32')
    else:
        ysh = None
    return Xsh, ysh
    