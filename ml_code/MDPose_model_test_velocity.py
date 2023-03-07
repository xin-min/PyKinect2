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
from dataloader_MDPose import velocityDataset
from datetime import datetime
import time
# import sys

# sys.stdout = open('outputlog_6Jan.txt','wt')
torch.manual_seed(42)

batch_size = 100
input_channels = 200 # doppler data at 1 time step 2x2400 data points
input_shape = (batch_size, 1, 4800) # doppler data
learning_rate = 0.05
output_size = 75           # 25 joints * 3 XYZ velocity :The output is prediction results for quaternion.  

device = ("cuda" if torch.cuda.is_available() else "cpu") 

train_dataloader = DataLoader(velocityDataset(), batch_size= batch_size, shuffle=False)

# [train_dataloader, val_dataloader, test_dataloader] = random_split(velocityDataset(), [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
# train_dataloader = DataLoader(train_dataloader, batch_size= batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataloader, batch_size= batch_size, shuffle=True)

# [train_dataloader, val_dataloader, test_dataloader] = random_split(dataset_LSTM_changequat(), [1, 0, 0], generator=torch.Generator().manual_seed(42))

# [train_dataloader, val_dataloader, test_dataloader] = random_split(dataset_LSTM(), [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))
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
	def __init__(self):#, input_size, output_size) x: 5*4800 y: 500 (flattened 5*100)
		super(Network, self).__init__() 
		self.cnn = nn.Sequential(
			# nn.BatchNorm1d(4800),
			nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'), 
			nn.BatchNorm1d(512), # batch norm
			nn.ReLU(),

			nn.Conv1d(512, 256, kernel_size=5, padding='same'),
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
			nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 2, batch_first = True),
			# nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),

			)

		self.linear = nn.Sequential(
			# nn.Linear(64, 6),
			# nn.ReLU(),
			# nn.Linear(25, 64),
			# nn.ReLU(),
			# nn.Linear(128, 75)
			nn.Linear(128, 93)

			)
		self.double()

	def forward(self, x): 
		# print(x.shape)
		x = torch.squeeze(x)
		x = torch.transpose(x,1,2)

		# x = torch.transpose(x,0,2)
		x1 = self.cnn(x) 
		# x1 = torch.transpose(x1,0,2)
		x1 = torch.transpose(x1,1,2)

		x2 = self.lstm(x1) 
		x2, (ht, ct) = self.lstm(x1)
		ct = ct[0]
		ct = ct[None,:]

		x3 = self.linear(ct)

		return x3


def test(): 
	# Load the model that we saved at the end of the training loop 
	torch.set_printoptions(profile="full")
	model = Network()
	model.to(device) 
	path = "huber_test_lstm_new_all.pt" 
	model.load_state_dict(torch.load(path)) 
	model.eval()
	 
	running_accuracy = 0 
	total = 0 

	print("TRAIN set now")

	f = open('huber_newlstm/IA_walk2.txt', 'w+')
	f1 = open('huber_newlstm/IA_walk2_truth.txt', 'w+')
	f2 = open('huber_newlstm/IA_walk2_kinect.txt', 'w+')

	total = 0
	correct = 0
	with torch.no_grad(): 
		# print(test_loader.size)
		for i, data in enumerate(train_dataloader, 0):
			train_inputs, train_labels, kinect_data = data
			train_inputs = torch.FloatTensor(train_inputs).double().cuda()
			train_inputs = train_inputs[None, :]
			train_labels = train_labels.cuda()
			train_labels = train_labels[None, :].double()
			train_labels = torch.transpose(train_labels,0,1)
			train_outputs = model(train_inputs)
			for y in train_outputs:
				f.write(str(y)+'\n')

			# train_outputs = [x/10-1 for x in y for y in train_outputs]
			# print(str(train_outputs)+'\n')
			# f.write(str(train_outputs)+'\n')
			for y in train_labels:
				f1.write(str(y)+'\n')

			# kinect_datas = np.array(kinect_data[0])
			# print(kinect_data.shape)
			# error
			temp = []
			for y in kinect_data:
				temp.append(torch.stack(y))
				# print(new.shape)
			kinect_data = torch.stack(temp)

			kinect_data = torch.transpose(kinect_data, 2, 0)
			kinect_data = torch.transpose(kinect_data, 1, 2)

			for y in kinect_data:
				f2.write(str(y)+'\n')

			# print(kinect_data.shape)
			# print(kinect_data[0])
			# error


			# # print(kinect_data)
			# print(len(kinect_data))
			# print(len(kinect_data[0]))
			# print(len(kinect_data[0][0]))

			# error
			# for y in kinect_data:
			# 	for x in y:
			# 		for z in x:
			# 			print(z)
			# 			f2.write(str(z)+'\n')
			# 	error


	f.close()
	f1.close()
	f2.close()




test()