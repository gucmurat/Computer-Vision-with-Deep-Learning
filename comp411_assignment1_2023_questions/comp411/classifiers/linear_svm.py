import numpy as np
from random import shuffle
import builtins


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

def huber_loss_naive(W, X, y, reg):
    """
    Modified Huber loss function, naive implementation (with loops).
    Delta in the original loss function definition is set as 1.
    Modified Huber loss is almost exactly the same with the "Hinge loss" that you have 
    implemented under the name svm_loss_naive. You can refer to the Wikipedia page:
    https://en.wikipedia.org/wiki/Huber_loss for a mathematical discription.
    Please see "Variant for classification" content.
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    ###############################################################################
    # TODO:                                                                       #
    # Complete the naive implementation of the Huber Loss, calculate the gradient #
    # of the loss function and store it dW. This should be really similar to      #
    # the svm loss naive implementation with subtle differences.                  #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
   # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score
            # If the margin > -1, calculate the margin loss and update gradients.
            # If the margin > 1, L = 4 * margin.
            # If the margin is within the range (-1, 1],  L = (1 + margin)^2.
            if margin > -1:
                if margin > 1:
                    loss += 4*margin
                    dW[:, j] += 4*X[i]
                    dW[:, y[i]] -= 4*X[i]
                else:
                    loss += (1+margin)**2
                    dW[:, j] += 2*(margin+1)*X[i]
                    dW[:, y[i]] -= 2*(margin+1)*X[i]
            
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X,W)
    
    # Compute the loss and the gradient by vectorized form rather than looping one by one.
    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
    margins[np.arange(num_train), y] = 0.0
    
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mask = (margins > 0).astype(float)
    mask[np.arange(num_train), y] -= np.sum(mask, axis = 1)
    dW += np.dot(X.T, mask)
    
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def huber_loss_vectorized(W, X, y, reg):
    """
    Structured Huber loss function, vectorized implementation.

    Inputs and outputs are the same as huber_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured Huber loss, storing the  #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    
    margins = np.zeros(scores.shape)
    margins_temp = scores - (scores[np.arange(num_train), y])[:, np.newaxis]
    
    # Compute the loss and the gradient by vectorized form rather than looping one by one.
    
    # If the margin > -1, calculate the margin loss and update gradients.
    # If the margin > 1, L = 4 * margin.
    # If the margin is within the range (-1, 1],  L = (1 + margin)^2.
    
    positive_mask = (margins_temp > 1)
    btw_mask = (-1 < margins_temp) & (margins_temp <= 1)
    margins[positive_mask] = 4*margins_temp[positive_mask]
    margins[btw_mask] = (margins_temp[btw_mask] + 1)**2
    margins[np.arange(num_train), y] = 0.0
    
    loss += np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured Huber     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # If the margin > -1, calculate the margin loss and update gradients.
    # If the margin > 1, mask = 4.
    # If the margin is within the range (-1, 1],  mask = (1 + margin)*2.
    mask = np.zeros(scores.shape)
    mask[positive_mask] = 4
    mask[btw_mask] = (margins_temp[btw_mask] + 1)*2
    mask[np.arange(num_train), y] = 0.0
    
    mask[np.arange(num_train), y] -= np.sum(mask,axis=1)

    dW += np.dot(X.T, mask)
    
    dW /= num_train
    dW += 2 * reg * W
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
