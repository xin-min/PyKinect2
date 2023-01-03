import os
import pandas as pd
# from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, labels_dir = "../data/8Dec/labels/250/"): #, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(labels_dir)
        self.labels_dir = labels_dir
        subfolders = ['rx', 'tx']
        for subfolder in subfolders:
            for file in os.listdir(self.labels_dir+subfolder):
                if file.endswith(".txt"):
                    files.append(file)
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
                doppler = [[float(x[0]), float(x[1])] for x in doppler]
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