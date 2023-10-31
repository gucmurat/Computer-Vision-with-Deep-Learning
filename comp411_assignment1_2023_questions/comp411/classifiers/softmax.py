import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for sample in range(num_train):
        scores = np.dot(X[sample], W)
        scores = np.exp(scores - np.max(scores))
        softmax = scores / np.sum(scores)
        loss -= np.log(softmax[y[sample]])
        for class_index in range(num_classes):
            dW[:, class_index] += (softmax[class_index] - (class_index == y[sample])) * X[sample]
            
    loss /= num_train
    dW /= num_train
    
    reg = reg_l2 * np.sum(W * W) + (regtype == 'ElasticNet') * reg_l1 * np.sum(np.abs(W))
    regdW = 2 * reg_l2 *  W + (regtype == 'ElasticNet') * reg_l1 * np.sign(W)
    
    dW += regdW
    loss += reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.dot(X, W)
    scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    scores /= np.sum(scores, axis=1, keepdims=True)
    loss += np.sum(-np.log(scores[np.arange(num_train), y])) / num_train
    scores[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, scores) / num_train
       
    reg = reg_l2 * np.sum(W * W) + (regtype == 'ElasticNet') * reg_l1 * np.sum(np.abs(W))
    regdW = 2 * reg_l2 *  W + (regtype == 'ElasticNet') * reg_l1 * np.sign(W)

    dW += regdW
    loss += reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
