import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
import math
import numpy as np
from kinect_quat_functions import abs2relquat
# from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['tx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            quat_file = pd.read_csv(lines[1][:-1])
            print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0]
                doppler = doppler_file.loc[:,doppler_time].values
                doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
                # print(len(doppler))
                try:
                    doppler = [[float(x[0]), float(x[1])] for x in doppler]
                except:
                    print(doppler)
                    continue
                # self.x_values.append(doppler)
                # print(doppler)
                # print(len(doppler))
                # print(len(doppler[0]))

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                for index in indexes:
                    index = int(index)
                    quat = quat_file.iloc[index].values[1:]
                    quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
                    self.x_values.append(torch.flatten(torch.Tensor(doppler))) # multiple quat values for the same doppler
                    self.y_values.append(torch.flatten(torch.Tensor(quat)))
                    
                    # print(quat)
                    # print(quat_file.iloc[index].values) 
                    # break


        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])



class CustomDataset_window(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['tx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            quat_file = pd.read_csv(lines[1][:-1])
            # print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0]
                doppler = doppler_file.loc[:,doppler_time].values
                doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
                # print(len(doppler))
                try:
                    doppler = [[float(x[0]), float(x[1])] for x in doppler]
                except:
                    print(doppler)
                    continue
                # self.x_values.append(doppler)
                # print(doppler)
                # print(len(doppler))
                # print(len(doppler[0]))

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                if len(indexes)<7:
                    continue

                new_indexes = []
                midpoint = math.floor(len(indexes)/2)
                for x in range(midpoint-2, midpoint+3): # middle 5 values
                    new_indexes.append(int(indexes[x]))

                y_quats = []

                for index in new_indexes:
                    quat = quat_file.iloc[index].values[1:]
                    quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
                    y_quats.append(quat)
                self.x_values.append(torch.flatten(torch.Tensor(doppler))) # multiple quat values for the same doppler
                self.y_values.append(torch.flatten(torch.Tensor(y_quats)))
                    
                    # print(quat)
                    # print(quat_file.iloc[index].values) 
                    # break


        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])


class CustomDataset_window_changequat(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['rx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            quat_file = pd.read_csv(lines[1][:-1])
            # print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0]
                doppler = doppler_file.loc[:,doppler_time].values
                doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
                # print(len(doppler))
                try:
                    doppler = [[float(x[0]), float(x[1])] for x in doppler]
                except:
                    print(doppler)
                    continue
                # self.x_values.append(doppler)
                # print(doppler)
                # print(len(doppler))
                # print(len(doppler[0]))

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                if len(indexes)<7:
                    continue

                new_indexes = []
                midpoint = math.floor(len(indexes)/2)
                for x in range(midpoint-3, midpoint+3): # middle 6 values (first value used to compute the 5 desired quaternions)
                    new_indexes.append(int(indexes[x]))

                temp_quats = []
                y_quats = []
                single_quat = []


                for index in new_indexes:
                    quat = quat_file.iloc[index].values[1:]
                    quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
                    temp_quats.append(quat)

                for index in range(5):
                    single_quat = []
                    for q in range(25):
                        quat1 = temp_quats[index + 1][q]
                        quat2 = temp_quats[index][q]
                        w0, x0, y0, z0 = quat1
                        modquat1 = math.sqrt(w0*w0 + x0*x0 + y0*y0 + z0*z0)
                        if modquat1 >0:
                            w0 = w0/modquat1
                            x0 = -x0/modquat1
                            y0 = -y0/modquat1
                            z0 = -z0/modquat1
                        w1, x1, y1, z1 = quat2
                        new_quat = [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0]
                        single_quat.append(new_quat) #q2*inv(q1)
                    y_quats.append(single_quat) #q2*inv(q1)




                # np.quaternion

                self.x_values.append(torch.flatten(torch.Tensor(doppler))) # multiple quat values for the same doppler
                self.y_values.append(torch.flatten(torch.Tensor(y_quats)))
                    
                    # print(quat)
                    # print(quat_file.iloc[index].values) 
                    # break


        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])



class CustomDataset_window_class(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['rx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            if "IAwalk" in file:
                class_num = torch.Tensor([1, 0, 0, 0, 0, 0]) #'walk' #0
            elif "IA_sit" in file:
                class_num = torch.Tensor([0, 1, 0, 0, 0, 0]) #'sit' #1
            elif "IA_SW" in file:
                class_num = torch.Tensor([0, 0, 1, 0, 0, 0]) #'SW' #2
            elif "IA_DW" in file:
                class_num = torch.Tensor([0, 0, 0, 1, 0, 0]) #'DW' #3
            elif "IA_Kick" in file:
                class_num = torch.Tensor([0, 0, 0, 0, 1, 0]) #'kick' #4
            elif "IA_Punch" in file:
                class_num = torch.Tensor([0, 0, 0, 0, 0, 1]) #'punch' #5
            else:
                # print("no haz")
                continue
            # print(str(class_num))
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            quat_file = pd.read_csv(lines[1][:-1])
            print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0]
                doppler = doppler_file.loc[:,doppler_time].values
                doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
                # print(len(doppler))
                try:
                    doppler = [[float(x[0]), float(x[1])] for x in doppler]
                except:
                    print(doppler)
                    continue
                # self.x_values.append(doppler)
                # print(doppler)
                # print(len(doppler))
                # print(len(doppler[0]))

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                if len(indexes)<7:
                    continue

                new_indexes = []
                midpoint = math.floor(len(indexes)/2)
                for x in range(midpoint-2, midpoint+3): # middle 5 values
                    new_indexes.append(int(indexes[x]))

                y_quats = []

                for index in new_indexes:
                    quat = quat_file.iloc[index].values[1:]
                    quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
                    y_quats.append(quat)
                self.x_values.append(torch.flatten(torch.Tensor(doppler))) # multiple quat values for the same doppler
                # self.y_values.append(torch.flatten(torch.Tensor(y_quats)))
                self.y_values.append(class_num)
                    # print(quat)
                    # print(quat_file.iloc[index].values) 
                    # break


        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])



class CustomDataset_class(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['rx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            if "IAwalk" in file:
                class_num = torch.Tensor([1, 0, 0, 0, 0, 0]) #'walk' #0
            elif "IA_sit" in file:
                class_num = torch.Tensor([0, 1, 0, 0, 0, 0]) #'sit' #1
            elif "IA_SW" in file:
                class_num = torch.Tensor([0, 0, 1, 0, 0, 0]) #'SW' #2
            elif "IA_DW" in file:
                class_num = torch.Tensor([0, 0, 0, 1, 0, 0]) #'DW' #3
            elif "IA_Kick" in file:
                class_num = torch.Tensor([0, 0, 0, 0, 1, 0]) #'kick' #4
            elif "IA_Punch" in file:
                class_num = torch.Tensor([0, 0, 0, 0, 0, 1]) #'punch' #5
            else:
                # print("no haz")
                continue
            # print(str(class_num))
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            # quat_file = pd.read_csv(lines[1][:-1])
            # print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0]
                doppler = doppler_file.loc[:,doppler_time].values
                doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
                # print(len(doppler))
                try:
                    doppler = [[float(x[0]), float(x[1])] for x in doppler]
                except:
                    print(doppler)
                    continue
                # self.x_values.append(doppler)
                # print(doppler)
                # print(len(doppler))
                # print(len(doppler[0]))

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                for index in indexes:
                    index = int(index)
                    # quat = quat_file.iloc[index].values[1:]
                    # quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    # quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
                    self.x_values.append(torch.flatten(torch.Tensor(doppler))) # multiple quat values for the same doppler
                    self.y_values.append(class_num)
                    
                    # print(quat)
                    # print(quat_file.iloc[index].values) 
                    # break


        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])



class dataset_LSTM(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['tx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            quat_file = pd.read_csv(lines[1][:-1])
            # print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0] # 1 doppler every 0.05s = 50 in doppler time. doppler time: HHMMSS---
                try:
                    temp_doppler = doppler_file.loc[:,str(int(doppler_time)-200):doppler_time].values # this should return 5*4800 (doppler at that time step + 4 time steps before it)
                except: # first few (4) time stamps in file
                    continue
                    #str(int(doppler_time)-200):
                if len(temp_doppler[0])<5:
                    continue
                temp_doppler = np.transpose(temp_doppler)
                doppler =[]

                for temp in temp_doppler:
                    temp = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in temp] # extracts all numbers, removes 'j' for the complex part of the number

                    try:
                        temp = [[float(x[0]), float(x[1])] for x in temp] # converts from complex to 2d float (x + yj) to (x, y)
                    except:
                        continue 

                    doppler.append(torch.flatten(torch.FloatTensor(temp)))
    

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                if len(indexes)<7:
                    continue

                new_indexes = []
                midpoint = math.floor(len(indexes)/2)
                for x in range(midpoint-2, midpoint+3): # middle 5 values 
                    new_indexes.append(int(indexes[x]))

                temp_quats = []
                y_quats = []
                single_quat = []


                for index in new_indexes:
                    quat = quat_file.iloc[index].values[1:]
                    quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
    
                    y_quats.append(quat)
                # print(y_quats)

                # for index in range(5):
                #     single_quat = []
                #     for q in range(25):
                #         quat1 = temp_quats[index + 1][q]
                #         quat2 = temp_quats[index][q]
                #         w0, x0, y0, z0 = quat1
                #         modquat1 = math.sqrt(w0*w0 + x0*x0 + y0*y0 + z0*z0)
                #         if modquat1 >0:
                #             w0 = w0/modquat1
                #             x0 = -x0/modquat1
                #             y0 = -y0/modquat1
                #             z0 = -z0/modquat1
                #         w1, x1, y1, z1 = quat2
                #         new_quat = [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                #          x1*w0 + y1*z0 - z1*y0 + w1*x0,
                #         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                #          x1*y0 - y1*x0 + z1*w0 + w1*z0]
                #         single_quat.append(new_quat) #q2*inv(q1)
                    # y_quats.append(single_quat) #q2*inv(q1)

                # np.quaternion
                self.x_values.append(torch.stack(doppler)) # multiple quat values for the same doppler


                # self.x_values.append(torch.FloatTensor(doppler)) # multiple quat values for the same doppler
                self.y_values.append(torch.flatten(torch.FloatTensor(y_quats)))

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])

class dataset_LSTM_changequat(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        # subfolders = ['rx', 'tx']
        subfolders = ['tx']
        files = []
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(self.labels_dir+subfolder+'/'+file)
        # print(files)
        # error

        files = [
        # '../data/8Dec/labels/250/tx/IAwalk1.txt', 
        # '../data/8Dec/labels/250/tx/IAwalk2.txt', 
        '../data/8Dec/labels/250/tx/IA_DW1.txt', 
        '../data/8Dec/labels/250/tx/IA_DW2.txt', 
        '../data/8Dec/labels/250/tx/IA_free2.txt', 
        '../data/8Dec/labels/250/tx/IA_free3.txt', 
        '../data/8Dec/labels/250/tx/IA_Kick1.txt', 
        '../data/8Dec/labels/250/tx/IA_Kick2.txt', 
        # '../data/8Dec/labels/250/tx/IA_pickup1.txt', 
        # '../data/8Dec/labels/250/tx/IA_pickup2.txt', 
        '../data/8Dec/labels/250/tx/IA_Punch1.txt', 
        '../data/8Dec/labels/250/tx/IA_Punch2.txt', 
        '../data/8Dec/labels/250/tx/IA_sit1.txt', 
        '../data/8Dec/labels/250/tx/IA_sit2.txt', 
        '../data/8Dec/labels/250/tx/IA_SW1.txt', 
        '../data/8Dec/labels/250/tx/IA_SW2.txt', 
        # '../data/8Dec/labels/250/tx/trevor_pickup1.txt', 
        # '../data/8Dec/labels/250/tx/trevor_pickup2.txt', 
        '../data/8Dec/labels/250/tx/trevor_sit1.txt', 
        '../data/8Dec/labels/250/tx/trevor_sit2.txt'
        # '../data/8Dec/labels/250/tx/trevor_walk1.txt', 
        # '../data/8Dec/labels/250/tx/trevor_walk2.txt'
        ]
        self.x_values = []
        self.y_values = []

        for file in files:
            f = open(file, "r")
            lines = f.readlines()

            doppler_file = pd.read_csv(lines[0][:-1])
            quat_file = pd.read_csv(lines[1][:-1])
            # print(lines[1][:-1])
            for x in range(2,len(lines)):
                line = lines[x].split(":")
                doppler_time = line[0] # 1 doppler every 0.05s = 50 in doppler time. doppler time: HHMMSS---
                try:
                    temp_doppler = doppler_file.loc[:,str(int(doppler_time)-200):doppler_time].values # this should return 5*4800 (doppler at that time step + 4 time steps before it)
                except: # first few (4) time stamps in file
                    continue
                    #str(int(doppler_time)-200):
                if len(temp_doppler[0])<5:
                    continue
                temp_doppler = np.transpose(temp_doppler)

                doppler =[]

                for temp in temp_doppler:
                    temp = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in temp] # extracts all numbers, removes 'j' for the complex part of the number
                    try:
                        temp = [[float(x[0]), float(x[1])] for x in temp] # converts from complex to 2d float (x + yj) to (x, y)
                    except:
                        continue 
                    doppler.append(torch.flatten(torch.FloatTensor(temp)))

                indexes = line[1].strip("\n").strip('][ ').split(', ')
                if len(indexes)<7:
                    continue

                new_indexes = []
                midpoint = math.floor(len(indexes)/2)
                for x in range(midpoint-3, midpoint+3): # middle 6 values 
                    new_indexes.append(int(indexes[x]))

                temp_quats = []
                y_quats = []
                single_quat = []


                for index in new_indexes:
                    quat = quat_file.iloc[index].values[1:]
                    quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
                    quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
                    temp_quats.append(quat)

                relative_quats = abs2relquat(temp_quats[0], temp_quats[1:])
                print(len(relative_quats))
                print(len(relative_quats[0]))
                error
                # for quat_num in range(1,len(temp_quats)):
                #     new_joint_quat = []
                #     quats = temp_quats[quat_num]
                #     prev_quat = temp_quats[quat_num-1] # current quat is quats
                #     for y in range(25):
                #         w0, x0, y0, z0 = prev_quat[y]
                #         w1, x1, y1, z1 = quats[y]
                #         modquat0 = math.sqrt(w0*w0 + x0*x0 + y0*y0 + z0*z0)

                #         if modquat0 >0:
                #             w0 = w0/modquat0
                #             x0 = -x0/modquat0
                #             y0 = -y0/modquat0
                #             z0 = -z0/modquat0
                        
                #         quat = [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                #          x1*w0 + y1*z0 - z1*y0 + w1*x0,
                #         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                #          x1*y0 - y1*x0 + z1*w0 + w1*z0]

                #         new_joint_quat.append(quat)
                #     y_quats.append(new_joint_quat)
                #     # print(len(new_joint_quat))


                # np.quaternion
                self.x_values.append(torch.stack(doppler)) # multiple quat values for the same doppler


                # self.x_values.append(torch.FloatTensor(doppler)) # multiple quat values for the same doppler
                self.y_values.append(torch.flatten(torch.FloatTensor(relative_quats)))
                

    def __len__(self): # number of samples in dataset
        return len(self.x_values)

    def __getitem__(self, index): 
        return (self.x_values[index], self.y_values[index])