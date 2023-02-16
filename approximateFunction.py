import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs
import scipy.io

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

steps=200
lr=1e-3
layers = np.array([1,50,50,20,50,50,1]) #5 hidden layers
min=0
max=2*np.pi
total_points=500
#Nu: Number of training points (2 as we onlt have 2 boundaries), # Nf: Number of collocation points (Evaluate PDE)
Nu=2
Nf=250

def f_BC(x): # This function satisfies the boundry conditions. The same as the real one (To ease the data generation), but we may not have it.
  return torch.sin(x)
def PDE(x): # The PDE equation. We use it to get the residual in the Neurl Network.
  return torch.cos(x)


class FCN(nn.Module):
    ##Neural Network
    def __init__(self, layers):
        super().__init__()  # call __init__ from parent class
        'activation function'
        self.activation = nn.Tanh()
        'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')
        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.iter = 0
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers) - 1):
            # weights from a normal distribution with
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)

            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    'foward pass'

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    'Loss Functions'
    # Loss BC
    def lossBC(self, x_BC):
        loss_BC = self.loss_function(self.forward(x_BC), f_BC(x_BC))
        return loss_BC

    # Loss PDE
    def lossPDE(self, x_PDE):
        g = x_PDE.clone()
        g.requires_grad = True  # Enable differentiation
        f = self.forward(g)
        f_x = autograd.grad(f, g, torch.ones([x_PDE.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        loss_PDE = self.loss_function(f_x, PDE(g))
        return loss_PDE

    def loss(self, x_BC, x_PDE):
        loss_bc = self.lossBC(x_BC)
        loss_pde = self.lossPDE(x_PDE)
        return loss_bc + loss_pde

    def closure(self):

        optimizer.zero_grad()

        loss = self.lossNN(x_train, y_train)

        loss.backward()

        self.iter += 1

        if self.iter % 100 == 0:
            print(loss)

        return loss

# get the analytical solution over the full domain
x = torch.linspace(min,max,total_points).view(-1,1) #prepare to NN
y = f_BC(x)
print(x.shape, y.shape)

fig, ax1 = plt.subplots()
ax1.plot(x.detach().numpy(), y.detach().numpy(), color='blue', label='Real_Train')
# ax1.plot(x_train.detach().numpy(),yh.detach().numpy(),color='red',label='Pred_Train')
ax1.set_xlabel('x', color='black')
ax1.set_ylabel('f(x)', color='black')
ax1.tick_params(axis='y', color='black')
ax1.legend(loc='upper left')

plt.show()

#def get_training_data(x):
#Nu: Number of training point, # Nf: Number of colloction points
# Set Boundary conditions:
BC_1=x[0,:]
BC_2=x[-1,:]
# Total Training points BC1+BC2
all_train=torch.vstack([BC_1,BC_2])
#Select Nu points
idx = np.random.choice(all_train.shape[0], Nu, replace=False)
x_BC=all_train[idx]
#Select Nf points
# Latin Hypercube sampling for collocation points
x_PDE = BC_1 + (BC_2-BC_1)*lhs(1,Nf)
x_PDE = torch.vstack((x_PDE,x_BC))

#Store tensors to GPU
torch.manual_seed(123)
x_PDE=x_PDE.float().to(device)
x_BC=x_BC.to(device)
#Create Model
model = FCN(layers)
print(model)
model.to(device)
params = list(model.parameters())
optimizer = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=False)
start_time = time.time()

for i in range(steps):
    yh = model(x_PDE)
    loss = model.loss(x_PDE,x_BC)# use mean squared error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%(steps/10)==0:
      print(loss)

yh=model(x.to(device))
y=f_BC(x)
#Error
print(model.lossBC(x.to(device)))

# Derivative
g=x.to(device)
g=g.clone()
g.requires_grad=True #Enable differentiation
f=model(g)
f_x=autograd.grad(f,g,torch.ones([g.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0]

# Detach from GPU
y_plot=y.detach().numpy()
yh_plot=yh.detach().cpu().numpy()
f_x_plot=f_x.detach().cpu().numpy()

# Plot
fig, ax1 = plt.subplots()
ax1.plot(x,y_plot,color='blue',label='Real')
ax1.plot(x,yh_plot,color='red',label='Predicted')
ax1.plot(x,f_x_plot,color='green',label='Derivative')
ax1.set_xlabel('x',color='black')
ax1.set_ylabel('f(x)',color='black')
ax1.tick_params(axis='y', color='black')
ax1.legend(loc = 'upper left')

plt.show()