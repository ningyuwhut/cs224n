#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import random
import sys

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    denominator=np.sqrt(np.sum(x**2,1)).reshape(-1,1)
    print "shape", denominator.shape
    x/=denominator

    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


#求当前单词的损失和对中心词和context中词的梯度
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec modelsW

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """
    ### YOUR CODE HERE
    print "predicted.shape", predicted.shape
    print "outputVectors.shape", outputVectors.shape
    print "target", target

    softmax_output = softmax(np.dot(outputVectors,predicted))

    print "softmax_output.shape", softmax_output.shape

    cost = -1 * np.log( softmax_output[target] )

    one_hot_y = np.zeros(outputVectors.shape[0])
    one_hot_y[target]=1
    print "one_hot_y", one_hot_y

    gradPred = np.dot(outputVectors.T, (softmax_output - one_hot_y).reshape(-1,1)).reshape(1,-1)

    grad = np.outer(softmax_output - one_hot_y,predicted)
     
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

   # print "target", target

    ### YOUR CODE HERE
   # print "indices", indices
   # print "outputVectors.shape", outputVectors.shape
   # print "predicted.shape", predicted.shape

    u0= outputVectors[target]

    s0= sigmoid(np.dot( u0, predicted))
  #  print "outputVecotrs", outputVectors
    neg_sample = outputVectors[indices[1:]]
 #   print "neg_sample", neg_sample

 #   print "neg_sample.shape", neg_sample.shape

    s = sigmoid(np.dot( neg_sample, predicted.reshape(predicted.shape[0],1)) )

  #  print "s", s
 #   print "s.shape", s.shape
    cost = -np.log(s0) - np.sum(np.log( 1- s))

 #   print "outputVectors[indices[1:]].T", outputVectors[indices[1:]].T.shape

    x=np.dot( neg_sample.T, s.reshape(-1,1) )

 #   print "x", x
 #   print "x.shape", x.shape

#    print "s0-1.0*u0", (s0-1.0)*u0
#    print "s0-1.0*u0", ((s0-1.0)*u0).shape

#    print "s0-1.0", s0-1.0

#这里不reshape的话就是 （1，3） 向量+ （3，1）向量，得到一个（3*3）向量

    gradPred = ((s0-1.0) * u0).reshape(-1,1) +  np.dot( neg_sample.T, s )

    grad = np.zeros( outputVectors.shape)

    grad[target] += (s0 -1.0)* predicted
#    print "indices", len(indices)
    for i in range(1, len(indices)):
        neg_sample_i = outputVectors[indices[i]]
        mu_k_v_c =  np.dot(neg_sample_i, predicted)
        s_i = sigmoid(mu_k_v_c)
        grad[indices[i]] +=   s_i  * predicted

   # print "tmp.shape", tmp.shape

    ### END YOUR CODE

    # grad = np.zeros(outputVectors.shape)
    # gradPred = np.zeros(predicted.shape)
    # cost = 0
    # z = sigmoid(np.dot(outputVectors[target], predicted))
    #
    # cost -= np.log(z)
    # grad[target] += predicted * (z - 1.0)
    # gradPred += outputVectors[target] * (z - 1.0)
    #
    # for k in xrange(K):
    #     samp = indices[k + 1]
    #     z = sigmoid(np.dot(outputVectors[samp], predicted))
    #     cost -= np.log(1.0 - z)
    #     grad[samp] += predicted * z
    #     gradPred += outputVectors[samp] * z
#    print "gradPred.type", gradPred.shape
#    print "grad.type", grad.shape
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
  #  print "currentWord", currentWord
    currentWordIndex = tokens[currentWord]
  #  print "currentWordIndex", currentWordIndex

    predicted = inputVectors[currentWordIndex]

    k=0
    for contextWord in contextWords:
 #       print "k in skip gram", k
        k+=1
 #       print "current contextWord", contextWord
        currentContextWordIndex = tokens[contextWord]
#        print "current context word index", currentContextWordIndex


        tmpCost, tmpGradIn,  tmpGradOut = word2vecCostAndGradient(predicted, currentContextWordIndex, outputVectors, dataset )

#        print "tmpGradIn.shape", tmpGradIn.shape
#        print "sadf", gradIn[currentWordIndex].shape
        cost += tmpCost
#        print "gradIn", gradIn
#        print "tmpGradIn", tmpGradIn
        gradIn[currentWordIndex] += tmpGradIn.reshape(len(tmpGradIn),)
#        print "gradIn", gradIn
        gradOut += tmpGradOut

    
    ### END YOUR CODE

    return cost, gradIn, gradOut

#由context 单词预测center word

#为什么加上平均之后就会校验失败

#在cbow 中input vector就是求和后的context word的向量，output vector就是中心词的vector

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    currentWordIndex = tokens[currentWord]
    contextWordSum = np.zeros(inputVectors.shape[1]) #inputVectors[currentWordIndex]

    for contextWord in contextWords:
        curCxtWordIndex = tokens[contextWord]
        contextWordSum += inputVectors[curCxtWordIndex]

    #contextWordSum /=len(contextWords)

    cost, tmpGradIn, gradOut = word2vecCostAndGradient(contextWordSum, currentWordIndex, outputVectors, dataset)

    for contextWord in contextWords:
        curCxtWordIndex = tokens[contextWord]

        a=tmpGradIn #.reshape(1,-1) #*len(contextWords)
        print "a", a
        print a.shape
        print gradIn[curCxtWordIndex].shape
        print gradIn.shape
        gradIn[curCxtWordIndex] += a.reshape(3,) #不明白为什么这里不加reshape(3,)就报错ValueError: non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (1,3)



    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

#tokens 是字典
#C是context的大小
#wordVectors 的行数是2*len(tokens)
#表示每个单词都有两个vector，一个是作为中心词的vector，一个是作为上下文词的vector

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
                         
 #   print "wordVectors", wordVectors
 #   print "wordVectors.shape", wordVectors.shape
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
#        print "sample index ", i
        C1 = random.randint(1,C) #随机生成一个context的大小
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


#
#dataset = type('dummy', (), {})()  创建了一个类型为dummy的类，具体参考
#https://docs.python.org/2/library/functions.html#type
def test_word2vec():
    """ Interface to the dataset for negative sampling """

    dataset = type('dummy', (), {})() 
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    print "dummy_vectors", dummy_vectors
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
  #  gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
  #      skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
  #      dummy_vectors)

    #sys.exit(1)
 #   gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
 #       skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
 #       dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    sys.exit(-1)
    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    #test_normalize_rows()
    test_word2vec()
