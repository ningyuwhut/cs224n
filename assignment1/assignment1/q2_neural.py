#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import random
import sys

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
#    print "params.shape", params.shape
#    print "params", str(params)

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1=np.dot(data,W1)+b1
    h=sigmoid(z1)
    z2=np.dot(h,W2)+b2
#    print "z2.shape ", z2.shape
    y_hat=softmax(z2) #M*Dy

#    print "y_hat", str(y_hat)
    cost = np.sum(-labels*np.log(y_hat))
    print "cost", str(cost)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    delta_1=y_hat-labels #M*Dy ,对z2的偏导
    delta_2=np.dot(delta_1,W2.T) #M*H, 对h的偏导

    delta_3=delta_2*sigmoid_grad(h) #M*H,对z1的偏导

    partial_x = np.dot(delta_3, W1.T)

    gradW2= np.dot(h.T, delta_1)
    gradb2= np.sum(delta_1, 0)

    gradW1 = np.dot(data.T, delta_3)
    gradb1=  np.sum(delta_3, 0)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum,生成一个N*dimensions[0]的随机矩阵
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
        #params是把所有的参数都放到一个向量里了

    print "params.shape", params.shape
    print "type(params)", type(params)
    print "params", str(params)

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
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
