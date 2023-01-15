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
### conv layer -> batch normalisation -> ooRELU (accelerate training speed and add non-linearity) -> final layer (output estimated values)

import numpy as np 
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from dataloader_MDPose import CustomDataset, CustomDataset_class, CustomDataset_window
from datetime import datetime
# import math
# import sys

# sys.stdout = open('outputlog_6Jan.txt','wt')
# import torch
torch.manual_seed(42)

batch_size = 1
input_channels = 4800 # doppler data at 1 time step 2x2400 data points
input_shape = (batch_size, 1, 2400) # doppler data
learning_rate = 0.01 
output_size = 100           # 25 joints * 4D quaternion :The output is prediction results for quaternion.  

device = ("cuda" if torch.cuda.is_available() else "cpu") 
f = open('outputlog_5epoch_poseest_huber.txt', 'a+')
PATH = "state_dict_model_outputlog_5epoch_poseest_huber.pt"

# train_dataloader = DataLoader(CustomDataset_class(), batch_size= batch_size, shuffle=True)
# print(str(torch.FloatTensor(train_dataloader).size))

# train_dataloader = DataLoader(CustomDataset_class(), batch_size= batch_size, shuffle=True)
[train_dataloader, val_dataloader, test_dataloader] = random_split(CustomDataset_window(), [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))


# print(str(torch.FloatTensor(train_dataloader).size))
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
			nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'),
			nn.BatchNorm1d(1), # batch norm
			nn.ReLU(),

			nn.Conv1d(512, 256, kernel_size=5, padding='same'),
			nn.BatchNorm1d(1), # batch norm
			nn.ReLU(),

			nn.Conv1d(256, 128, kernel_size=5, padding='same'),
			nn.BatchNorm1d(1), # batch norm
			nn.ReLU(),

			nn.Conv1d(128, 64, kernel_size=5, padding='same'),
			nn.BatchNorm1d(1), # batch norm
			nn.ReLU(),



			# nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'),
			# nn.BatchNorm1d(1), # batch norm
			# nn.ReLU(),


			# nn.Conv1d(512, 256, kernel_size=5, padding='same'),
			# nn.BatchNorm1d(1), # batch norm
			# nn.ReLU(),


			# nn.Conv1d(256, 128, kernel_size=5, padding='same'),
			# nn.BatchNorm1d(1), # batch norm
			# nn.ReLU(),

			# nn.Conv1d(128, 64, kernel_size=5, padding='same'),
			# nn.BatchNorm1d(1), # batch norm
			# nn.ReLU(),

			# nn.Conv1d(64, 32, kernel_size=5, padding='same'),
			# nn.BatchNorm1d(1), # batch norm
			# nn.ReLU(),

			nn.Flatten()
			)

		self.lstm = nn.Sequential(
			nn.LSTM(input_size = 64, hidden_size = 25, num_layers = 2),
			# nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),

			)

		self.linear = nn.Sequential(
			# nn.Linear(64, 6),
			# nn.ReLU(),
			nn.Linear(64, 100),
			nn.ReLU(),
			nn.Linear(100, 500)
			)

	def forward(self, x): 
	   x1 = self.cnn(x) 
	   x1 = torch.transpose(x1,0,1)
	   # print(x1.shape)
	   # x2 = self.lstm(x1) 
	   # x2, (ht, ct) = self.lstm(x1)
	   # print(x2.shape)
	   # x2 = x2[0]
	   # print(x2.shape)
	   x3 = self.linear(x1)

	   return x3

# Instantiate the model 
criterion = nn.HuberLoss()
model = Network() 
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# f = open('outputlog_7Jan.txt', 'a+')

for epoch in range(5):  # loop over the dataset multiple times

	running_loss = 0.0
	
	for i, data in enumerate(train_dataloader, 0):
		
		inputs, labels = data
		inputs = inputs.cuda()
		inputs = inputs[None, :]
		labels = labels.cuda()
		labels = labels[None, :]

		# print(inputs.shape)
		# print(labels.shape)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(torch.transpose(inputs,0,1))
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		# if i % 2000 == 1999:    # print every 5000 mini-batches
		# 	now = datetime.now()
		# 	print(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
		# 	f.write(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}'+'\n')
		# 	running_loss = 0.0
	now = datetime.now()
	print(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.5f}')
	f.write(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.5f}'+'\n')
	# running_loss = 0.0
	
	val_loss = 0
	for i, data in enumerate(val_dataloader, 0):
		val_inputs, val_labels = data
		val_inputs = val_inputs.cuda()
		val_inputs = val_inputs[None, :]
		val_labels = val_labels.cuda()
		val_labels = val_labels[None, :]
		val_outputs = model(torch.transpose(val_inputs,0,1))
		val_loss += criterion(val_outputs, val_labels)
		largest = i
	print(f'val_loss = {val_loss / largest:.3f} \n' )

print('Finished Training')
f.close()

torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))
model.eval()

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






