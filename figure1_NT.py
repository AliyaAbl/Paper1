
from __future__ import print_function
import math
import numpy as np
import scipy.linalg as scl
import scipy.sparse as ss
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import tensorflow as tf
from   scipy.fftpack import dct, idct
import math
import torch.optim as optim
import time
from   torch.utils.data import Dataset, DataLoader




def NTK2(X, Z):
    """This function computes NTK kernel for two-layer ReLU neural networks via
    an analytic formula.

    Input:
    X: d times n_1 matrix, where d is the feature dimension and n_i are # obs.
    Z: d times n_2 matrix, where d is the feature dimension and n_i are # obs.

    output:
    C: The kernel matrix of size n_1 times n_2.
    """

    print(np.shape(X)[0], np.shape(Z)[0])
    pi = math.pi
    assert X.shape[0] == Z.shape[0]
    # X is sized d \times n
    nx = np.linalg.norm(X, axis=0, keepdims=True)
    nx = nx.T    
    nz = np.linalg.norm(Z, axis=0, keepdims=True)    

    C = np.dot(X.T, Z) #n_1 * n_2
    C = np.multiply(C, (nx ** -1))
    C = np.multiply(C, (nz ** -1))
    # Fixing numerical mistakes
    C = np.minimum(C, 1.0)
    C = np.maximum(C, -1.0)			

    C = np.multiply(1.0 - np.arccos(C) / pi, C) + np.sqrt(1 - np.power(C, 2)) / (2 * pi)
    C = np.multiply(nx, np.multiply(C, nz))
    return C

def compute_kernel(X_train, X_test, Y_train, Y_test):
    """This function computes the test and the training kernels.
    Inputs:
        name: Kernel name.
        hyper_param: Kernel hyper-parameters.
    Outputs:
        Training Kernel: n times n np.float32 matrix.
        Test Kernel: nt times n np.float32 matrix.
        ytrain: vector of training labels. n times 1 np.float32.
        ytest: vector of test labels. nt times 1 np.float32.
    """	
    X = X_train
    Xtest = X_test	
    K = NTK2(X.T, X.T)
    KT = NTK2(Xtest.T, X.T)

    return (K, KT, Y_train, Y_test)

def compute_accuracy(true_labels, preds):
	"""This function computes the classification accuracy of the vector
	preds. """
	if true_labels.shape[1] == 1:
		n = len(true_labels)
		true_labels = true_labels.reshape((n, 1))
		preds = preds.reshape((n, 1))		
		preds = preds > 0
		inds = true_labels > 0
		return np.mean(preds == inds)	
	groundTruth = np.argmax(true_labels, axis=1).astype(np.int32)
	predictions = np.argmax(preds, axis=1).astype(np.int32)
	return np.mean(groundTruth == predictions)


########
# From Arda
def getFilter(dim):
  F = np.zeros((dim,dim))
  for i in range(dim):
    for j in range(dim):
      if ((dim -i)**2+(dim - j)**2) <= ((dim - 1)**2):
        F[i,j] = 1.0
  return F

def get_data_with_HF_noise(tau,x_train_,y_train, plot):
  """
  Input:  tau = noise strength
          x_train = FMNIST train dataset
          y_train = FMNIST train labels
  Output: X_noisy = Dataset with added noise
          Y_train = labels 
  """
  # implement 2D DCT
  def dct2(a):
      return dct(dct(a.T, norm='ortho').T, norm='ortho')

  # implement 2D IDCT
  def idct2(a):
      return idct(idct(a.T, norm='ortho').T, norm='ortho')   

  # Generate plain fmnist data and preprocessing
  N = x_train_.shape[0]     # nr of samples
  d = x_train_.shape[1]     # Dimension of input image
  d_flatten = d**2          # flattened dimension of input image
  x_train  = x_train_.flatten().reshape(N, d**2)

  F = getFilter(d)          # Filter which frequencies should get noise
  X_noisy = np.zeros((x_train.shape[0],d,d))
  for i in range(N):        # for every image
    Z = np.random.randn(F.shape[0],F.shape[1])
    Z_tilde = np.multiply(Z,F)  # Hadmard product => noise matrix
    img = x_train[i,:]
    img = img - np.mean(img)    #Remove global mean to "center" data
    img = img.reshape((d, d))
    img_freq_space = dct2(img)  # now we got the frequencies of the image. Next step add noise!
    img_noisy_freq = img_freq_space + tau* (np.linalg.norm(img_freq_space)/np.linalg.norm(Z_tilde))*Z_tilde #See 3. of appendix
    img_noisy = idct2(img_noisy_freq)                               # transform back to pixel space by inverse discrete fourier
    img_noisy = img_noisy /np.linalg.norm(img_noisy) * math.sqrt(d) # normalize to norm sqrt(d)
    X_noisy[i,:,:] = img_noisy
  
  if plot == False:
    X_noisy = (torch.from_numpy(X_noisy.flatten().reshape(N, d_flatten))).float()
  
  Y_train = (torch.from_numpy(y_train)).long()
  return X_noisy, Y_train

class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, data,label):
        """Method to initilaize variables.""" 
        
        self.labels = label
        self.images = data

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index,:]
        
        return image, label

    def __len__(self):
        return len(self.images)


########
expand = True
mean = 0.1
reg = 1e-3
tau = np.linspace(0, 3, 15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

(x_train_, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
history_tau = []
tau = np.linspace(0,3,num=15) # 15 points for different noises in their plot; Noise strength
for i in range(len(tau)):
    
    (x_train_, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train,Y_train = get_data_with_HF_noise(tau=tau[i],x_train_=x_train_,y_train=y_train, plot=False)
    X_test, Y_test  = get_data_with_HF_noise(tau=tau[i],x_train_=x_test,  y_train=y_test, plot=False)

    errors = np.zeros((4)) #Train Loss, Train Accuracy, Test Loss, Test Accuracy
    print('Computing Kernels...')
    K, KT, ytrain, ytest = compute_kernel(X_train, X_test, Y_train, Y_test)
    print('Done computing Kernels.')

    n = len(ytrain)
    nt = len(ytest)
    if expand:
        print('expanding labels...')
        maxVal = np.int(ytrain.max())
        Y = np.zeros((n, maxVal + 1), dtype=np.float32)
        Y[np.arange(n), ytrain[:, 0].astype(np.int32)] = 1.0

        YT = np.zeros((nt, maxVal + 1), dtype=np.float32)
        YT[np.arange(nt), ytest[:, 0].astype(np.int32)] = 1.0
        Y = Y - mean
        YT = YT - mean
        assert Y.dtype == np.float32
        assert YT.dtype == np.float32
        ytrain = Y
        ytest = YT
        mean = 0.0

    RK = K + reg * np.eye(n, dtype=np.float32)	
    assert K.dtype == np.float32
    assert RK.dtype == np.float32
    print('Solving kernel regression with %d observations and regularization param %f'%(n, reg))
    t1 = time.time()
    if expand:		
        x = scl.solve(RK, ytrain, assume_a='sym')
    else:
        cg = ss.linalg.cg(RK, ytrain[:, 0] - mean, maxiter=400, atol=1e-4, tol=1e-4)
        x = np.copy(cg[0]).reshape((n, 1))
    t2 = time.time()
    print('iteration took %f seconds'%(t2 - t1))
    yhat = np.dot(K, x) + mean
    preds = np.dot(KT, x) + mean
    errors[0] = np.linalg.norm(ytrain - yhat) ** 2 / (len(ytrain) + 0.0)
    errors[2] = np.linalg.norm(ytest - preds) ** 2 / (len(ytest) + 0.0)
    errors[1] = compute_accuracy(ytrain - mean, yhat - mean)
    errors[3] = compute_accuracy(ytest - mean, preds - mean)

    history_tau = np.append(history_tau, errors[3])

  