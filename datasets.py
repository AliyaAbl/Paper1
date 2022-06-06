import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from   scipy.fftpack import dct, idct
import math
import torch.optim as optim
import time
from   torch.utils.data import Dataset, DataLoader


def generate_nonuniform_data(n, d, eta, kappa):
  """
  Input:  n     = number of samples 
          d     = total dimensions
          eta   = fraction of signal dimension
          kappa = ???
  Output: X_train   = [n,      d]
          X_test    = [10 000, d]
          Y_train   = [n,      1]
          Y_test    = [10 000, 1]
  """

  ########## define the parameters ######################
  #######################################################
  d1        = int(d ** eta)            # d_0 = d_{signal}
  nt        = 10000                       # number of test samples
  exponent  = (eta + kappa) / 2.0         # kappa / 2
  r         = d ** exponent               # snr = r = r_1/r_2

  ########## generate training data #####################
  #######################################################
  np.random.seed(145)
  X = np.random.normal(size=(n, d))
  X = X.astype(np.float32)
  for i in range(n):
      X[i, :d1] = X[i, :d1] / np.linalg.norm(X[i, :d1]) * r
      X[i, d1:] = X[i, d1:] / np.linalg.norm(X[i, d1:]) * np.sqrt(d)
      
  ########## generate testing data ######################
  #######################################################
  np.random.seed(2)
  XT = np.random.normal(size=(nt, d))
  XT = XT.astype(np.float32)
  for i in range(nt):
      XT[i, :d1] = XT[i, :d1] / np.linalg.norm(XT[i, :d1]) * r
      XT[i, d1:] = XT[i, d1:] / np.linalg.norm(XT[i, d1:]) * np.sqrt(d)
      
  X_train = X
  X_test  = XT

  ########## remove ambient dimensions ##################
  #######################################################
  X0 = X[:, :d1]
  X1 = XT[:, :d1]
  del X, XT

  ########## generate labels for training data ##########
  #######################################################
  beta2 = np.random.exponential(scale=1.0, size=(d1 - 1, 1))
  beta3 = np.random.exponential(scale=1.0, size=(d1 - 2, 1))
  beta4 = np.random.exponential(scale=1.0, size=(d1 - 3, 1))
  
  ########## generate labels for training data ##########
  #######################################################
  np.random.seed(14)
  f = []
      
  Z = np.multiply(X0[:, :-1], X0[:, 1:])
  temp = np.dot(Z, beta2)
  f.append(temp)

  Z = np.multiply(X0[:, :-2], X0[:, 1:-1])
  Z = np.multiply(Z, X0[:, 2:])
  temp = np.dot(Z, beta3)
  f.append(temp)

  Z = np.multiply(X0[:, :-3], X0[:, 1:-2])
  Z = np.multiply(Z, X0[:, 2:-1])
  Z = np.multiply(Z, X0[:, 3:])
  temp = np.dot(Z, beta4)
  f.append(temp)
  
  normalization = [np.sqrt(np.mean(t ** 2)) for t in f]
  for i in range(len(f)):
      f[i] = f[i] / normalization[i]
      
  totalf = f[0] + f[1] + f[2]
  totalf = totalf.astype(np.float32)

  Y_train = totalf
  
  ########## generate labels for testing data ###########
  #######################################################
  g = []
  
  Z = np.multiply(X1[:, :-1], X1[:, 1:])
  temp = np.dot(Z, beta2)
  g.append(temp)

  Z = np.multiply(X1[:, :-2], X1[:, 1:-1])
  Z = np.multiply(Z, X1[:, 2:])
  temp = np.dot(Z, beta3)
  g.append(temp)

  Z = np.multiply(X1[:, :-3], X1[:, 1:-2])
  Z = np.multiply(Z, X1[:, 2:-1])
  Z = np.multiply(Z, X1[:, 3:])
  temp = np.dot(Z, beta4)
  g.append(temp)
  for i in range(len(g)):
      g[i] = g[i] / normalization[i]
  totalg = g[0] + g[1] + g[2]
  totalg = totalg.astype(np.float32)

  Y_test  = totalg

  return X_train, X_test, Y_train, Y_test
    
    
d     = 1024
eta   = 2/5   # d_signal = 16
kappa = np.linspace(0, 0.9, 10)
n     =  2**20 
nt    = 10**4

########## generate 10 datasets for different values of kappa ##########
########################################################################
X_train1,  X_test1,  Y_train1,  Y_test1  = generate_nonuniform_data(n, d, eta, kappa[0])
X_train2,  X_test2,  Y_train2,  Y_test2  = generate_nonuniform_data(n, d, eta, kappa[1])
X_train3,  X_test3,  Y_train3,  Y_test3  = generate_nonuniform_data(n, d, eta, kappa[2])
X_train4,  X_test4,  Y_train4,  Y_test4  = generate_nonuniform_data(n, d, eta, kappa[3])
X_train5,  X_test5,  Y_train5,  Y_test5  = generate_nonuniform_data(n, d, eta, kappa[4])
X_train6,  X_test6,  Y_train6,  Y_test6  = generate_nonuniform_data(n, d, eta, kappa[5])
X_train7,  X_test7,  Y_train7,  Y_test7  = generate_nonuniform_data(n, d, eta, kappa[6])
X_train8,  X_test8,  Y_train8,  Y_test8  = generate_nonuniform_data(n, d, eta, kappa[7])
X_train9,  X_test9,  Y_train9,  Y_test9  = generate_nonuniform_data(n, d, eta, kappa[8])
X_train10, X_test10, Y_train10, Y_test10 = generate_nonuniform_data(n, d, eta, kappa[9])
