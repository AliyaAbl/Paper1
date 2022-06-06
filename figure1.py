import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import tensorflow as tf
from   scipy.fftpack import dct, idct
import math
import torch.optim as optim
import time
import torchvision
from   torch.utils.data import Dataset, DataLoader


# From Arda's Notebook 
class NeuralNetwork_FMNIST(nn.Module):
  """
  This is the 2-layerd fully-connected neuronal network with K hidden neurons and 1 output neuron, used thoughtout this report. 
  """

  def __init__(self,K):
    """
    Input:  K                         = number of hidden neurons
            N                         = number of samples            
    """
    print("Creating a Neural Network ")
    super(NeuralNetwork_FMNIST, self).__init__()
        
    self.g    = nn.ReLU()
    self.soft = nn.Softmax()
    self.K    = K
    self.loss = nn.MSELoss(reduction='mean')
    self.fc1  = nn.Linear(28*28, K, bias=False)
    self.fc2  = nn.Linear(K, 10, bias=False)
    nn.init.normal_(self.fc1.weight)
    nn.init.normal_(self.fc2.weight)

  def forward(self, x):
    x = self.fc1(x)
    x = self.g(x)
    x = self.fc2(x)
    x = self.soft(x)
    return x

def train(model, loss_fn, train_data, val_data, epochs=750, device='cpu'):

    print('train() called: model=%s, epochs=%d, device=%s\n' % \
          (type(model).__name__, epochs, device))

    history = {} 
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec  = time.time()
    l2_reg          = 1e-4
    lr_0            = 1e-3

    for epoch in range(1, epochs+1):
        if epoch <=15:
          # 15 linear warm-up epochs in the beginning
          train_dl = DataLoader(train_data, batch_size=500,shuffle=True)
          val_dl   = DataLoader(val_data,   batch_size=500,shuffle=True)
        else: 
          # After "Warm up" increase batch size to 1000!
          train_dl = DataLoader(train_data, batch_size=1000,shuffle=True)
          val_dl   = DataLoader(val_data,   batch_size=1000,shuffle=True)

        # Learning rate evolution
        lr_t = lr_0 * np.max([1+np.cos(epoch*np.pi/epochs),1/15])
        optimizer = optim.SGD(model.parameters(), lr=lr_t, momentum=0.9,weight_decay=l2_reg)

        ######## Training ###################################################
        #####################################################################
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()
            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)

            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)

        ######## Evaluating ##################################################   
        ######################################################################        
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        if epoch == 1 or epoch % 10 == 0: #show progress every 10 epochs
          print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP

    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

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


########## define parameters ##########################
#######################################################

tau = np.linspace(0, 3, 15)

#Dataset

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#Get and load data into dataloader
(x_train_, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
history_tau = []
tau = np.linspace(0,3,num=15) # 15 points for different noises in their plot; Noise strength
for i in range(len(tau)):
  net = NeuralNetwork_FMNIST(K=4096).to(device)
  criterion = nn.CrossEntropyLoss()
  print("Tau={}".format(tau[i]))
  print("Generate Data with noise in high frequencies....")
  X_train,Y_train = get_data_with_HF_noise(tau=tau[i],x_train_=x_train_,y_train=y_train, plot=False)
  X_test, Y_test  = get_data_with_HF_noise(tau=tau[i],x_train_=x_test,  y_train=y_test, plot=False)
  train_data = FashionDataset(X_train,Y_train)
  val_data   = FashionDataset(X_test, Y_test)
  print("Start training....")
  history = train(
      model = net,
      loss_fn = criterion,
      device=device,
      train_data = train_data,
      val_data = val_data)
  history_tau.append(history["val_acc"])


np.save('history_tau.npy', history_tau)