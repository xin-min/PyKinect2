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
from dataloader_MDPose import velocityDataset
from datetime import datetime
# import math
# import sys

# sys.stdout = open('outputlog_6Jan.txt','wt')
# import torch
torch.manual_seed(42)

batch_size = 100
input_channels = 200 # doppler data at 1 time step 2x2400 data points
input_shape = (batch_size, 1, 2400) # doppler data
learning_rate = 0.01
output_size = 100           # 25 joints * 4D quaternion :The output is prediction results for quaternion.  
num_epochs = 10000

error  

device = ("cuda" if torch.cuda.is_available() else "cpu") 
f = open('huber_test_lstm_0703_all.txt', 'a+')
PATH = "huber_test_lstm_0703_all.pt"

# train_dataloader = DataLoader(CustomDataset_class(), batch_size= batch_size, shuffle=True)
# print(str(torch.FloatTensor(train_dataloader).size))

train_dataloader = DataLoader(velocityDataset(), batch_size= batch_size, shuffle=False)
# [train_dataloader, val_dataloader, test_dataloader] = random_split(velocityDataset(), [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
# train_dataloader = DataLoader(train_dataloader, batch_size= batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataloader, batch_size= batch_size, shuffle=True)


class Network(nn.Module): 
	# input to conv1d: (N, C, L)
	# N is a batch size, C denotes a number of channels, L is a length of signal sequence

	def __init__(self):#, input_size, output_size) x: 5*4800 y: 500 (flattened 5*100)
		super(Network, self).__init__() 
		self.cnn = nn.Sequential(
			# nn.BatchNorm1d(4800), # batch norm



			nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'), 
			# nn.MaxPool1d(3,2,1),
			nn.BatchNorm1d(512), # batch norm
			nn.ReLU(),

			nn.Conv1d(512, 256, kernel_size=5, padding='same'),
			# nn.MaxPool1d(3,2,1),

			nn.BatchNorm1d(256), # batch norm
			nn.ReLU(),

			nn.Conv1d(256, 128, kernel_size=5, padding='same'),
			nn.BatchNorm1d(128), # batch norm
			nn.ReLU(),

			# nn.Conv1d(128, 64, kernel_size=5, padding='same'),
			# nn.BatchNorm1d(64), # batch norm
			# nn.ReLU(),
			
			# nn.Flatten()
			)

		# shape of input to lstm model: [batch_size, seq_len, input_size], input_size = number of features
		self.lstm = nn.Sequential(
			# nn.LSTM(input_size = 128, hidden_size = 25, num_layers = 2, batch_first = True),
			nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 2, batch_first = True),

			# nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),

			)

		self.linear = nn.Sequential(
			# nn.Linear(64, 6),
			# nn.ReLU(),
			# nn.Linear(25, 64),
			# nn.ReLU(),
			# nn.Linear(128, 75)
			nn.Linear(128, 93) # tree structure

			)
		self.double()

	def forward(self, x):
	   # print(x.shape)

	   # x = torch.transpose(x,1,2)
	   # print(x.shape)
	   # x = torch.transpose(x,0,2)


	   # x = torch.transpose(x,0,2)
	   # x = torch.transpose(x,1,2)
	   x = torch.squeeze(x)
	   # print(x.shape)
	   x = torch.transpose(x,1,2)#.float()
	   # x = torch.transpose(x,0,1)

	   # print(x.type)


	   x1 = self.cnn(x) 
	   # # print(x1.shape)

	   # x1 = torch.transpose(x1,0,2)
	   x1 = torch.transpose(x1,1,2)

	   # print(x1.shape)

	   # print(x1.shape)
	   x2 = self.lstm(x1) 
	   x2, (ht, ct) = self.lstm(x1)
	   ct = ct[0]
	   ct = ct[None,:]
	   ct = torch.transpose(ct,0,1)
	   # print(ct.shape)


	   x3 = self.linear(ct)
	   # print(x3.shape)


	   return x3

# Instantiate the model 
criterion = nn.HuberLoss()
# criterion = nn.MSELoss()

model = Network() 
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# f = open('outputlog_7Jan.txt', 'a+')

for epoch in range(num_epochs):  # loop over the dataset multiple times

	running_loss = 0.0
	
	for i, data in enumerate(train_dataloader, 0):
		# print(data.shape)
		(inputs, labels, nil)= data
		# print(len(inputs))#.shape)
		# print(len(inputs[0]))
		# print(labels.shape)


		# inputs = torch.Tensor(torch.flatten(inputs)).cuda()
		# inputs = torch.flatten(torch.FloatTensor(inputs)).cuda()
		# inputs = torch.FloatTensor(inputs).cuda()
		# print(len(inputs))
		# print(len(inputs[0]))
		# print(len(inputs[0][0]))


		# ERROR
		# print(inputs)
		# error
		# inputs = np.array(inputs, dtype='float')
		inputs = torch.FloatTensor(inputs).double().cuda()
		# inputs = torch.stack(inputs).double().cuda()

		# inputs = torch.FloatTensor(inputs)
		# inputs = inputs.cuda()
		inputs = inputs[None, :]
		labels = torch.Tensor(labels).cuda()
		labels = labels[None, :].double()
		labels = torch.transpose(labels,0,1)
		# labels = torch.transpose(labels,1,2)

		# print(inputs.shape)
		# print(labels.shape)

		# # zero the parameter gradients
		# optimizer.zero_grad()

		# forward + backward + optimize
		# print(inputs.shape)
		outputs = model(inputs)#torch.transpose(inputs,0,1))
		# print(outputs)
		# print(outputs.size)
		# print(outputs)
		# print(labels)
		loss = criterion(outputs, labels)

		# zero the parameter gradients
		optimizer.zero_grad()
		
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
	if (epoch+1)%100 ==0:
		# print(outputs)
		print(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.5f}')
		f.write(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.5f}'+'\n')
		
	# running_loss = 0.0
	
	# val_loss = 0
	# for i, data in enumerate(val_dataloader, 0):
	# 	val_inputs, val_labels = data
	# 	val_inputs = val_inputs.cuda()
	# 	val_inputs = val_inputs[None, :]
	# 	val_labels = val_labels.cuda()
	# 	val_labels = val_labels[None, :]
	# 	val_outputs = model(torch.transpose(val_inputs,0,1))
	# 	val_loss += criterion(val_outputs, val_labels)
	# 	largest = i
	# print(f'val_loss = {val_loss / largest:.5f} \n' )
# print(outputs)

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






