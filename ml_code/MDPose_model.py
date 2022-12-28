# use conda environment: 

# MDPose_model
### CNN-RNN architecture
### 1-d conv layer over each frame - learn spatial features
### Feed concatenated features into 2-layer LSTM RNN - learn time dependency
### Pass output of each time step through fully-connected layers - generate estimated sequence

# data
### inputs: chnage in quaternion


# Conv layer
### kernel size: 5x5 (fine-grained feature extraction)
### conv layer -> batch normalisation -> RELU (accelerate training speed and add non-linearity) -> final layer (output estimated values)

import numpy as np 
import torch as th
from torch import nn

batch_size = ?
input_channels = 1 # doppler data at 1 time step 1x2400 data points
input_shape = (batch_size, 1, 2400) # doppler data
learning_rate = 0.01 
output_size = 100           # 25 joints * 4D quaternion :The output is prediction results for quaternion.  

device = ("cuda" if torch.cuda.is_available() else "cpu") 


self.cnn = nn.Sequential(
    nn.Conv1d(input_channels, conv1_output, kernel_size=5, padding='same'),
    nn.BatchNorm1D(conv1_output) # batch norm
    nn.ReLU()


    nn.Conv1d(32, 64, kernel_size=5, padding='same'),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=5, stride=5),
    nn.Conv1d(64,128, kernel_size=5, padding='same'),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=4, stride=4),
    # nn.ReLU(),
    nn.Flatten(),
)

# Compute shape by doing one forward pass
with th.no_grad():
    n_flatten = self.cnn(
        th.as_tensor(observation_space.sample()["laserscan"]).float().reshape((1,1,360))
    ).shape[1]
    # print(n_flatten)
    # if len(n_flatten) < 3:
    #     n_flatten = n_flatten[0] * n_flatten[1]

self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
self.total = nn.Sequential(self.cnn,self.linear)




# Define neural network 
class Network(nn.Module): 
   def __init__(self, input_size, output_size): 
       super(Network, self).__init__() 
        
       self.layer1 = nn.Linear(input_size, 24) 
       self.layer2 = nn.Linear(24, 24) 
       self.layer3 = nn.Linear(24, output_size) 


   def forward(self, x): 
       x1 = F.relu(self.layer1(x)) 
       x2 = F.relu(self.layer2(x1)) 
       x3 = self.layer3(x2) 
       return x3 
 
# Instantiate the model 
model = Network(input_size, output_size) 