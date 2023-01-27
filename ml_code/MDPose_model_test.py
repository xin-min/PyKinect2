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
from dataloader_MDPose import dataset_LSTM_changequat, dataset_LSTM, CustomDataset, CustomDataset_class, CustomDataset_window
from datetime import datetime
import time
# import sys

# sys.stdout = open('outputlog_6Jan.txt','wt')
torch.manual_seed(42)

batch_size = 1
input_channels = 4800 # doppler data at 1 time step 2x2400 data points
input_shape = (batch_size, 1, 2400) # doppler data
learning_rate = 0.01 
output_size = 100           # 25 joints * 4D quaternion :The output is prediction results for quaternion.  

device = ("cuda" if torch.cuda.is_available() else "cpu") 

train_dataloader = DataLoader(dataset_LSTM_changequat(), batch_size= batch_size, shuffle=False)
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
            nn.LSTM(input_size = 64, hidden_size = 25, num_layers = 2, batch_first = True),
            # nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),

            )

        self.linear = nn.Sequential(
            # nn.Linear(64, 6),
            # nn.ReLU(),
            nn.Linear(25, 100),
            nn.ReLU(),
            nn.Linear(100, 500)
            )

    def forward(self, x): 
       x1 = x[:,:,0,:]
       x1 = torch.transpose(x1,0,1)
       x1 = torch.transpose(x1,1,2)


       x1 = self.cnn(x1) 
       # print(x1.shape)

       x1 = torch.transpose(x1,1,2)
       # print(x1.shape)
       # x2 = self.lstm(x1) 
       x2, (ht, ct) = self.lstm(x1)
       # print(x2.shape)
       # x2 = x2[0]
       # print(x2.shape)
       x3 = self.linear(x2)

       return x3

    # def __init__(self):#, input_size, output_size):
    #     super(Network, self).__init__() 
    #     self.cnn = nn.Sequential(
    #         nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'),
    #         nn.BatchNorm1d(1), # batch norm
    #         nn.ReLU(),

    #         nn.Conv1d(512, 256, kernel_size=5, padding='same'),
    #         nn.BatchNorm1d(1), # batch norm
    #         nn.ReLU(),

    #         nn.Conv1d(256, 128, kernel_size=5, padding='same'),
    #         nn.BatchNorm1d(1), # batch norm
    #         nn.ReLU(),

    #         nn.Conv1d(128, 64, kernel_size=5, padding='same'),
    #         nn.BatchNorm1d(1), # batch norm
    #         nn.ReLU(),

    #         # nn.Conv1d(input_channels, 512, kernel_size=5, padding='same'),
    #         # nn.BatchNorm1d(1), # batch norm
    #         # nn.ReLU(),

    #         # nn.Conv1d(512, 256, kernel_size=5, padding='same'),
    #         # nn.BatchNorm1d(1), # batch norm
    #         # nn.ReLU(),

    #         # nn.Conv1d(256, 128, kernel_size=5, padding='same'),
    #         # nn.BatchNorm1d(1), # batch norm
    #         # nn.ReLU(),

    #         # nn.Conv1d(128, 64, kernel_size=5, padding='same'),
    #         # nn.BatchNorm1d(1), # batch norm
    #         # nn.ReLU(),

    #         # nn.Conv1d(64, 32, kernel_size=5, padding='same'),
    #         # nn.BatchNorm1d(1), # batch norm
    #         # nn.ReLU(),

    #         nn.Flatten()
    #         )

    #     self.lstm = nn.Sequential(
    #         nn.LSTM(input_size = 64, hidden_size = 25, num_layers = 2),
            
    #         # nn.LSTM(input_size = 32, hidden_size = 5, num_layers = 2),
    #         )

    #     self.linear = nn.Sequential(
    #         # nn.Linear(64, 6),
    #         # nn.ReLU(),
    #         nn.Linear(64, 100),
    #         nn.ReLU(),
    #         nn.Linear(100, 500)
    #         # nn.Linear(25, 100)
    #         )

    # def forward(self, x): 
    #    x1 = self.cnn(x) 
    #    x1 = torch.transpose(x1,0,1)
    #    # print(x1.shape)
    #    # x2 = self.lstm(x1) 
    #    # x2, (ht, ct) = self.lstm(x1)
    #    # print(x2.shape)
    #    # x2 = x2[0]
    #    # print(x2.shape)
    #    x3 = self.linear(x1)

    #    return x3


def test(): 
    # Load the model that we saved at the end of the training loop 
    torch.set_printoptions(profile="full")
    model = Network()
    model.to(device) 
    path = "state_dict_model_outputlog_new_joint_quat.pt" 
    model.load_state_dict(torch.load(path)) 
    model.eval()
     
    running_accuracy = 0 
    total = 0 

    print("TRAIN set now")

    f = open('state_dict_model_outputlog_new_joint_quat_newtest.txt', 'a+')
    total = 0
    correct = 0
    with torch.no_grad(): 
        # print(test_loader.size)
        for i, data in enumerate(train_dataloader, 0):
            train_inputs, train_labels = data
            train_inputs = train_inputs.cuda()
            train_inputs = train_inputs[None, :]
            train_labels = train_labels.cuda()
            train_labels = train_labels[None, :]
            train_outputs = model(torch.transpose(train_inputs,0,1))
            # print(str(train_outputs)+'\n')
            f.write(str(train_outputs)+'\n')

    #         predicted_value = int(torch.argmax(train_outputs))
    #         truth = int(torch.argmax(train_labels))

    #         if truth ==predicted_value:
    #             correct +=1
    #         total +=1
    # print("TRAINSET total: "+str(total) + "    correct: "+str(correct))
    # print("val set now")
    f.close()

    # f = open('state_dict_model_outputlog_70epoch_poseest_huberrx_LSTM_val.txt', 'a+')


    # total = 0
    # correct = 0
    # with torch.no_grad(): 
    #     # print(test_loader.size)
    #     for i, data in enumerate(val_dataloader, 0):
    #         val_inputs, val_labels = data
    #         val_inputs = val_inputs.cuda()
    #         val_inputs = val_inputs[None, :]
    #         val_labels = val_labels.cuda()
    #         val_labels = val_labels[None, :]
    #         val_outputs = model(torch.transpose(val_inputs,0,1))
    #         # print(str(val_outputs)+'\n')
    #         f.write(str(val_outputs)+'\n')


    # #         predicted_value = int(torch.argmax(val_outputs))
    # #         truth = int(torch.argmax(val_labels))

    # #         if truth ==predicted_value:
    # #             correct +=1
    # #         total +=1
    # # print("VALSET total: "+str(total) + "    correct: "+str(correct))

    # f.close()

    # f = open('state_dict_model_outputlog_70epoch_poseest_huberrx_LSTM_test.txt', 'a+')

    # print("test set now")
    # total = 0
    # correct = 0
    # with torch.no_grad(): 
    #     # print(test_loader.size)
    #     for i, data in enumerate(test_dataloader, 0):
    #         test_inputs, test_labels = data
    #         test_inputs = test_inputs.cuda()
    #         test_inputs = test_inputs[None, :]
    #         test_labels = test_labels.cuda()
    #         test_labels = test_labels[None, :]
    #         test_outputs = model(torch.transpose(test_inputs,0,1))
    #         # print(str(test_outputs)+'\n')
    #         f.write(str(test_outputs)+'\n')
    # f.close()

    #         predicted_value = int(torch.argmax(test_outputs))
    #         truth = int(torch.argmax(test_labels))

    #         if truth ==predicted_value:
    #             correct +=1
    #         total +=1
    # print("TESTSET total: "+str(total) + "    correct: "+str(correct))


    #     val_loss += criterion(val_outputs, val_labels)
    #     largest = i
    # print(f'val_loss = {val_loss / largest:.3f} \n' )

    #     for data in test_dataloader: 
    #         inputs, outputs = data 
    #         outputs = outputs.to(torch.float32) 
    #         inputs = inputs.cuda()
    #         outputs = outputs.cuda()
    #         # print(str(outputs))
    #         # print(str(torch.argmax(outputs)))
    #         truth = int(torch.argmax(outputs))
    #         predicted_outputs = model(torch.transpose(inputs,0,1)) 
    #         # f.write(str(predicted_outputs)+'\n')
    #         # print(str(predicted_outputs)+'\n')
    #         # print(str(torch.argmax(predicted_outputs))+'\n')
    #         predicted_value = int(torch.argmax(predicted_outputs))
    #         if truth ==predicted_value:
    #             correct +=1
    #         total +=1
    #         if total%5000 ==0:
    #             print(str(total))
    #         if total>20000:
    #             break
    # print("total: "+str(total)+'\n' + "correct: "+str(correct))

            # time.sleep(2)
            # _, predicted = torch.max(predicted_outputs, 1) 
            # total += outputs.size(0) 
            # running_accuracy += (predicted == outputs).sum().item() 
 
        # print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))    
 
test()