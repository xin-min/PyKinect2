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
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataloader_MDPose import CustomDataset

batch_size = 4
input_channels = 1 # doppler data at 1 time step 1x2400 data points
input_shape = (batch_size, 1, 2400) # doppler data
learning_rate = 0.01 
output_size = 100           # 25 joints * 4D quaternion :The output is prediction results for quaternion.  

device = ("cuda" if torch.cuda.is_available() else "cpu") 

train_dataloader = DataLoader(CustomDataset(), batch_size= batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)



# Define neural network 
class Network(nn.Module): 

	def __init__(self):#, input_size, output_size):
		super(Network, self).__init__() 
		self.cnn = nn.Sequential(
			nn.Conv1d(input_channels, 128, kernel_size=5, padding='same'),
			nn.BatchNorm1d(128), # batch norm
			nn.ReLU(),

			nn.Conv1d(128, 64, kernel_size=5, padding='same'),
			nn.BatchNorm1d(64), # batch norm
			nn.ReLU(),

			nn.Conv1d(64, 32, kernel_size=5, padding='same'),
			nn.BatchNorm1d(32), # batch norm
			nn.ReLU(),

			nn.Flatten()
			)

		self.lstm = nn.Sequential(
			nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 100)
			)

	def forward(self, x): 
	   x1 = self.cnn(x) 
	   x2 = self.lstm(x1) 
	   return x2
 
# Instantiate the model 
criterion = nn.MSELoss()
model = Network() 
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(train_dataloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
			running_loss = 0.0

print('Finished Training')

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# print()

# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

# PATH = "state_dict_model.pt"
# torch.save(model.state_dict(), PATH)
# model.load_state_dict(torch.load(PATH))
# model.eval()






