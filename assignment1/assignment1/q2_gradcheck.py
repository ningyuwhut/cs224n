#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import random


#检验时要注意x是作为f的参数的，f是一个函数指针
#所以每次在一个点修改了x后需要再改回去
#f返回的是损失函数，是一个标量
# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    print "x " + str(x)
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
#        print "x[" + str(ix) + "]=" + str(x[ix])

#        print "x.shape" + str(x.shape)
#        print "type(x)" + str(type(x))
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
#        print len(x.shape)
        x[ix]+=h
        print "ix", str(ix)
        print "x", str(x)
        random.setstate(rndstate)
        fx_plus_h, _ = f(x)

        x[ix]-=(2*h)
        random.setstate(rndstate)
        fx_minus_h, _ = f(x)
        x[ix]+=h

        numgrad = (fx_plus_h - fx_minus_h)/(2*h)


        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            return
        print "Your gradient: %f \t Numerical gradient: %f" % (
            grad[ix], numgrad)

        it.iternext() # Step to next dimension

    print "Gradient check passed!"

    print "x", str(x)

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    #gradcheck_naive(quad, np.array(123.456))      # scalar test
    #gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
