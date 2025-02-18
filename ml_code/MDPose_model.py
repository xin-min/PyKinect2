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
from torch.utils.data import DataLoader, random_split
from dataloader_MDPose import dataset_LSTM, CustomDataset, CustomDataset_class, CustomDataset_window, CustomDataset_window_changequat, dataset_LSTM_changequat
from datetime import datetime
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
# import math
# import sys

# sys.stdout = open('outputlog_6Jan.txt','wt')
# import torch
torch.manual_seed(42)

batch_size = 100
input_channels = 200 # doppler data at 1 time step 2x2400 data points
input_shape = (batch_size, 1, 2400) # doppler data
learning_rate = 0.03
output_size = 100           # 25 joints * 4D quaternion :The output is prediction results for quaternion.  
num_epochs = 80000
random_seed = 42
# error

device = ("cuda" if torch.cuda.is_available() else "cpu") 
f = open('IA_walk1_80000.txt', 'a+')
PATH = "IA_walk1_80000.pt"

# train_dataloader = DataLoader(CustomDataset_class(), batch_size= batch_size, shuffle=True)
# print(str(torch.FloatTensor(train_dataloader).size))


train_dataloader = DataLoader(dataset_LSTM_changequat(), batch_size= 100, shuffle=False)

# ############### split train-test
# dataset = dataset_LSTM_changequat()
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# shuffle_dataset = True
# test_split = .2
# split = int(np.floor(test_split * dataset_size))
# if shuffle_dataset :
# 	np.random.seed(random_seed)
# 	np.random.shuffle(indices)
# train_indices, test_indices = indices[split:], indices[:split]

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)

# train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
# 										   sampler=train_sampler)
# test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
# 										   sampler=test_sampler)

# ############### split train-test



# [train_dataloader, val_dataloader, test_dataloader] = random_split(dataset_LSTM_changequat(), [0.8, 0, 0.2], generator=torch.Generator().manual_seed(42))


class Network(nn.Module): 
	# input to conv1d: (N, C, L)
	# N is a batch size, C denotes a number of channels, L is a length of signal sequence

	def __init__(self):#, input_size, output_size) x: 5*4800 y: 500 (flattened 5*100)
		super(Network, self).__init__() 
		self.cnn = nn.Sequential(
			nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'), 
			nn.BatchNorm1d(512), # batch norm
			nn.ReLU(),

			nn.Conv1d(512, 256, kernel_size=5, padding='same'),
			nn.BatchNorm1d(256), # batch norm
			nn.ReLU(),

			nn.Conv1d(256, 128, kernel_size=5, padding='same'),
			nn.BatchNorm1d(128), # batch norm
			nn.ReLU(),

			nn.Conv1d(128, 64, kernel_size=5, padding='same'),
			nn.BatchNorm1d(64), # batch norm
			nn.ReLU(),
			
			# nn.Flatten()
			)

		# shape of input to lstm model: [batch_size, seq_len, input_size], input_size = number of features
		self.lstm = nn.Sequential(
			# nn.LSTM(input_size = 128, hidden_size = 25, num_layers = 2, batch_first = True),
			nn.LSTM(input_size = 64, hidden_size = 25, num_layers = 2, batch_first = True),

			# nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),

			)

		self.linear = nn.Sequential(
			# nn.Linear(64, 6),
			# nn.ReLU(),
			# nn.Linear(25, 64),
			# nn.ReLU(),
			nn.Linear(64, 100)
			)

	def forward(self, x):
	   # print(x.shape)
	   
	   # x = x[0]
	   x = torch.transpose(x,0,1)
	   x = torch.transpose(x,1,2)
	   # print(x.shape)


	   x1 = self.cnn(x) 
	   # print(x1.shape)

	   x1 = torch.transpose(x1,1,2)
	   # print(x1.shape)
	   # x2 = self.lstm(x1) 
	   # x2, (ht, ct) = self.lstm(x1)
	   # print(x2.shape)
	   # x2 = x2[0]
	   # print(x2.shape)
	   x3 = self.linear(x1)
	   # print(x3.shape)
	   # error

	   return x3

# Instantiate the model 
criterion = nn.HuberLoss()
model = Network() 
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)


# f = open('outputlog_7Jan.txt', 'a+')

# ############### split train-test

# running_loss_graph = []
# valid_loss_graph = []
# c_graph=0

# ############### split train-test


for epoch in range(num_epochs):  # loop over the dataset multiple times

	running_loss = 0.0
	
	for i, data in enumerate(train_dataloader, 0):
		# print(data.shape)
		inputs, labels = data
		# print(inputs.shape)
		# print(labels.shape)


		inputs = inputs.cuda()
		inputs = inputs[None, :]
		labels = labels.cuda()
		# labels = labels[None, :]
		# labels = labels[0]


		# print(inputs.shape)
		# print(labels.shape)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		# print(outputs.size)
		# print(outputs.shape)
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

	# ################ split train-test

	# valid_loss = 0.0
	# model.eval()     # Optional when not using Model Specific layer
	
	# for j, data in enumerate(test_dataloader, 0):
	# 	# print(j)
	# 	inputs, labels = data
	# 	# print(inputs.shape)
	# 	# print(labels.shape)


	# 	inputs = inputs.cuda()
	# 	inputs = inputs[None, :]
	# 	labels = labels.cuda()
	# 	# if torch.cuda.is_available():
	# 	# 	data, labels = data.cuda(), labels.cuda()
		
	# 	target = model(inputs)
	# 	loss = criterion(target,labels)
	# 	valid_loss += loss.item()# * i.size(0)

	# ################ split train-test

	


	now = datetime.now()
	if (epoch+1)%100==0:
		
		print(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.5f}')
		f.write(str(now) + f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.5f}'+'\n')

	# ################ split train-test

	# 	print(str(now) + f'[{epoch + 1}, {i + 1:5d}] val_loss: {valid_loss / (j+1):.5f}')
	# 	f.write(str(now) + f'[{epoch + 1}, {i + 1:5d}] val_loss: {valid_loss / (j+1):.5f}'+'\n')


	# 	running_loss_graph.append(running_loss / (i+1))
	# 	valid_loss_graph.append(valid_loss / (j+1))
	# 	c_graph+=1
	# ################ split train-test



	# scheduler.step()
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

print('Finished Training')
f.close()

torch.save(model.state_dict(), PATH)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# ################ split train-test

# x = range(c_graph)
# # print([running_loss_graph,valid_loss_graph])
# df = pd.DataFrame ([running_loss_graph,valid_loss_graph]).transpose()
# # pd.options.plotting.backend = "plotly"
# df.columns = ['running loss', 'val loss']
# fig = df.plot(backend = "plotly")
# fig.show()
# fig.write_image("images/fig1.png")
# # time.sleep(10)

# ################ split train-test





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






