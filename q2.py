# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def calculate_a(test_datum, x_train, tau):
    constant = 2*tau*tau
    # test_datum is dx1 and x_train is Nxd
    # function l2 expects second dim of both to be the same
    # need to transpose test_datum
    # NOTE: l2 needs test_datum to be in a column format as it 
    # accesses axis=1, reshape.
    test_datum = np.reshape(test_datum, (1, test_datum.shape[0]))
    sqnorm = l2(test_datum, x_train)
    scaled_norm = -sqnorm/constant
    
    # We're supposed to calculate e(N)/sum(e(D)) but now we have e(N) and log(sum(e(D)))
    # We can do the followimg: 
    # N/D = e^log(e(N)/sum(e(D))) = e^(log(e(N))-log(sum(e(D)))
    #     = e(N - log(sum(e(D))))
    denuminator_log = scipy.special.logsumexp(scaled_norm)
    a = np.exp(scaled_norm - denuminator_log)
    return a

def get_optimal_weights(x_train, x_transpose, y_train, a, lam):
    A = np.diag(a.reshape(a.shape[1]))
    
    # Wmin = Inv(XT A X + lambdaI) XT A Y
    # We need to solve this for W, linalg.solve needs something of the form AW = B
    # The inverse part of the term above can be "uninverted" and used as the term A
    # The second part can be term B.
    termA = np.matmul(x_transpose, A)
    termA = np.matmul(termA, x_train)
    lambdaI = lam * np.identity(termA.shape[0])
    termA = termA + lambdaI

    termB = np.matmul(x_transpose, A)
    termB = np.matmul(termB, y_train)

    weights = np.linalg.solve(termA, termB)
    return weights

def predict(x, weights):
    y = np.matmul(x, weights)
    return y

#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    x_transpose = x_train.transpose()
    a = calculate_a(test_datum, x_train, tau)
    w = get_optimal_weights(x_train, x_transpose, y_train, a, lam)
    y = predict(test_datum, w)
    return y

def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    # shuffle and split the x and y, we want x and y to shuffle exactly the same way
    rng_state = np.random.get_state()           # Get state of RNG
    np.random.shuffle(x)
    np.random.set_state(rng_state)              # Reset state of RNG so y shuffling would match x shuffling
    np.random.shuffle(y)
    last_index = int((1-val_frac)*N)
    x_train = x[:last_index]
    x_test = x[last_index:]
    y_train = y[:last_index]
    y_test = y[last_index:]

    train_losses = np.array([])
    test_losses = np.array([])
    for tau in taus:
        y_hats = np.array([])
        for data in x_train:
            y_hat = LRLS(data, x_train, y_train, tau)
            y_hats = np.append(y_hats, y_hat)
        errors = y_hats - y_train
        train_losses = np.append(train_losses, np.mean(errors**2))
        
        y_hats = np.array([])
        for data in x_test:
            y_hat = LRLS(data, x_train, y_train, tau)
            y_hats = np.append(y_hats, y_hat)
        errors = y_hats - y_test
        test_losses = np.append(test_losses, np.mean(errors**2))
        print('At tau=' + str(round(tau, 2)) + ': train_loss=' + str(round(train_losses[-1], 2)) + ', test_loss=' + str(round(test_losses[-1], 2)), end='\r')
    return train_losses, test_losses

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.xlabel("tau")
    plt.ylabel("Train loss")
    plt.show()
    plt.semilogx(test_losses)
    plt.xlabel("tau")
    plt.ylabel("Test loss")
    plt.show()

